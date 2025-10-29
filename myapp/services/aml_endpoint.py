# myapp/services/aml_endpoint.py
import os, json, re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from .text_normalizer import normalize_df_kana, normalize_columns

try:
    from azure.identity import DefaultAzureCredential
except Exception:
    DefaultAzureCredential = None


# ====== 設定クラス ======
@dataclass
class AMLConfig:
    scoring_uri: str
    template_json_path: str
    auth_mode: str = os.getenv("AML_AUTH_MODE", "key")  # "key" | "aad"
    api_key: Optional[str] = os.getenv("AML_API_KEY")
    drop_columns: str = os.getenv("DROP_COLUMNS", "")
    timeout_sec: int = int(os.getenv("AML_TIMEOUT", "60"))
    connect_timeout_sec: int = int(os.getenv("AML_CONNECT_TIMEOUT", "10"))
    batch_size: int = int(os.getenv("AML_BATCH_SIZE", "256"))
    deployment: Optional[str] = os.getenv("AML_DEPLOYMENT")  # optional header
    prediction_col: str = os.getenv("ML_PREDICTION_COL", "prediction")
    limit_rows: int = int(os.getenv("AML_LIMIT_ROWS", "0"))

    @staticmethod
    def from_env() -> "AMLConfig":
        scoring_uri = os.getenv("AML_SCORING_URI")
        template = os.getenv("TEMPLATE_JSON")
        if not scoring_uri:
            raise RuntimeError("AML_SCORING_URI が未設定です。")
        if not template or not os.path.exists(template):
            raise RuntimeError("テンプレJSONが見つかりません。.env に TEMPLATE_JSON=./template.json を設定し、Studio Test タブの JSON を保存してください。")
        return AMLConfig(scoring_uri=scoring_uri, template_json_path=template)

# 日付/時間らしさの列名ヒントを拡充
_DATE_COL_HINTS = (
    "日", "日付", "年月日", "date", "Date", "DATE",
    "日時", "time", "Time", "TIME",
    "時間", "時刻",
    "引渡", "納期", "期限"
)

# ====== ヘルパ ======
def _normalize_cols_list(s: str) -> List[str]:
    return [c.strip().strip("\"").strip("'") for c in s.split(",") if c.strip()]


def _load_template(template_path: str) -> Tuple[Dict[str, Any], List[str]]:
    with open(template_path, "r", encoding="utf-8") as f:
        tpl = json.load(f)
    try:
        cols = tpl["input_data"]["columns"]
        if not isinstance(cols, list) or not cols:
            raise KeyError
        return tpl, cols
    except Exception:
        raise RuntimeError("テンプレJSONに 'input_data.columns' が見つかりません。Testタブの columns を含む JSON を保存してください。")

def _normalize_colname(name: str) -> str:
    """
    列名の表記ゆれを吸収するための正規化:
      - NFKC正規化（全角/半角統一）
      - 前後空白の除去
      - 半角カナ → 全角（NFKCでほぼ吸収）
      - 記号の微差は基本そのまま（「-」「_」等は保持）
    """
    if name is None:
        return ""
    s = unicodedata.normalize("NFKC", str(name))
    return s.strip()


def _reorder_like_template(df: pd.DataFrame, tpl_columns: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    template.json の columns を基準に、名前ベースで df を並べ替える。
    - 双方の列名に _normalize_colname を適用して突き合わせ
    - 見つからないテンプレ列は None で埋める
    - 余剰列はドロップ（AzureMLに送らない）
    返り値: (整列後DF, 不足テンプレ列, 破棄した余剰列)
    """
    # 正規化マップ
    df_norm_map = {_normalize_colname(c): c for c in df.columns}
    tpl_norm = [_normalize_colname(c) for c in tpl_columns]

    ordered_series = []
    missing = []
    for raw_tpl, norm_tpl in zip(tpl_columns, tpl_norm):
        if norm_tpl in df_norm_map:
            ordered_series.append(df[df_norm_map[norm_tpl]])
        else:
            missing.append(raw_tpl)
            ordered_series.append(pd.Series([None] * len(df), index=df.index))

    # 余剰列の検出（テンプレに存在しない列）
    tpl_norm_set = set(tpl_norm)
    extras = [orig for norm, orig in df_norm_map.items() if norm not in tpl_norm_set]

    out = pd.concat(ordered_series, axis=1)
    out.columns = tpl_columns
    return out, missing, extras

def _shape_df_for_template(df: pd.DataFrame, expected_cols: List[str], drop_cols_csv: str) -> pd.DataFrame:
    # 明示ドロップ（目的変数など）
    drops = _normalize_cols_list(drop_cols_csv)
    for c in drops:
        if c in df.columns:
            df = df.drop(columns=[c])

    missing = [c for c in expected_cols if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected_cols]
    if missing or extra:
        # ログ用途（ビュー側で messages に流用可）
        print(">>> 列の差分チェック")
        if missing: print("  - 期待にあるがCSVに無い列:", missing[:30], "...")
        if extra:   print("  - CSVにあるが期待に無い列:", extra[:30], "...")

    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols]


def _to_jsonable(v):
    if pd.isna(v): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if hasattr(v, "isoformat"):
        try: return v.isoformat()
        except: pass
    return v


def _payload_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    data = [[_to_jsonable(v) for v in row] for row in df.to_numpy()]
    return {"input_data": {"columns": df.columns.tolist(), "data": data}}

_NUMERIC_LIKE_RE = re.compile(
    r"^\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*$|^\s*[-+]?\d+(?:\.\d+)?\s*$"
)

def _coerce_numeric_like_series(s: pd.Series) -> pd.Series:
    # object/string のみ対象
    if s.dtype.kind not in ("O", "U", "S"):
        return s
    def _one(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return x
        xs = str(x).strip()
        if _NUMERIC_LIKE_RE.match(xs):
            xs = xs.replace(",", "")
            try:
                if xs.isdigit() or (xs.startswith(("-", "+")) and xs[1:].isdigit()):
                    return int(xs)
                return float(xs)
            except Exception:
                return x
        return x
    return s.map(_one)

def _coerce_datetime_like_series(name: str, s: pd.Series) -> pd.Series:
    # 列名ヒントに反応、かつ object/string のみ対象
    if not any(h in name for h in _DATE_COL_HINTS):
        return s
    if s.dtype.kind not in ("O", "U", "S"):
        return s
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        return dt
    except Exception:
        return s

def _sanitize_values_for_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 1) 数値らしき文字列 → 数値へ
    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        out[c] = _coerce_numeric_like_series(out[c])
    # 2) 日付/時間らしき列名 → datetime へ（_to_jsonable が isoformat にしてくれる）
    for c in out.columns:
        out[c] = _coerce_datetime_like_series(c, out[c])
    return out


def _get_bearer(auth_mode: str, api_key: Optional[str]) -> str:
    if auth_mode == "key":
        if not api_key:
            raise RuntimeError("API_KEY が未設定です（AUTH_MODE=key）。")
        return api_key
    # AAD
    if DefaultAzureCredential is None:
        raise RuntimeError("azure-identity が必要です（pip install azure-identity）。")
    cred = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    return cred.get_token("https://ml.azure.com/.default").token


def _post(endpoint: str, bearer: str, payload: Dict[str, Any], connect_timeout: int, read_timeout: int, deployment: Optional[str]) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer}",
    }
    if deployment:
        headers["azureml-model-deployment"] = deployment
    return requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=(connect_timeout, read_timeout))

# ====== メイン: エンドポイント推論 ======
def score_via_endpoint(df: pd.DataFrame, cfg: AMLConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    DEBUG_PAYLOAD_PREVIEW = os.getenv("AML_DEBUG_PAYLOAD_PREVIEW", "0") in ("1","true","True")
    DEBUG_SHOW_SAMPLE_COL = os.getenv("AML_DEBUG_SAMPLE_COL", "ユニットID")

    df = normalize_df_kana(df.copy())
    df.columns = normalize_columns(list(df.columns))

    # テンプレはヘルパで取得（input_data.columns を保証）
    _, expected_cols_raw = _load_template(cfg.template_json_path)  # テンプレそのまま（表示/送信用）
    expected_cols_norm = normalize_columns(expected_cols_raw)      # 比較用に正規化

    # ① DROP_COLUMNS（正規化してから適用）
    if cfg.drop_columns:
        drops_raw = _normalize_cols_list(cfg.drop_columns)
        drops = set(normalize_columns(drops_raw))
        if drops:
            expected_cols_norm = [c for c in expected_cols_norm if c not in drops]
            # テンプレ“生”配列からも同じインデックスを落とす（順序保持のため）
            expected_cols_raw = [raw for raw, norm in zip(expected_cols_raw, normalize_columns(expected_cols_raw)) if norm not in drops]
        drop_targets = [c for c in df.columns if c in drops]
        if drop_targets:
            df = df.drop(columns=drop_targets)
            print(f">>> 明示ドロップ: {drop_targets}")

    # ★テンプレ順へ“名前ベース”再整列
    #   - df.columns と expected_cols_norm はどちらも normalize_columns 済み
    #   - mapping: 正規化名 → 現在の実カラム名
    norm_to_actual = {c: c for c in df.columns}
    ordered_actual_cols = []
    missing_norm = []
    for norm in expected_cols_norm:
        if norm in norm_to_actual:
            ordered_actual_cols.append(norm_to_actual[norm])
        else:
            missing_norm.append(norm)
            ordered_actual_cols.append(None)  # 後で埋める

    # 代表的な差分ログ
    extra_norm = [c for c in df.columns if c not in set(expected_cols_norm)]
    if missing_norm or extra_norm:
        print(">>> 列の差分チェック(正規化後)")
        # 生のテンプレ名も見たいので、missing は raw でも表示
        missing_raw = [raw for raw, norm in zip(expected_cols_raw, normalize_columns(expected_cols_raw)) if norm in set(missing_norm)]
        print(f"  - 期待にあるがCSVに無い列(raw): {missing_raw}")
        print(f"  - CSVにあるが期待に無い列(norm): {extra_norm}")

    # 欠損列を None で補間しつつテンプレ順の DataFrame を作る
    ordered_frames = []
    for col in ordered_actual_cols:
        if col is None:
            ordered_frames.append(pd.Series([None]*len(df), index=df.index))
        else:
            ordered_frames.append(df[col])
    df = pd.concat(ordered_frames, axis=1)
    # 送信用の見た目として“テンプレの生カラム名”をセット（デバッグもしやすい）
    df.columns = expected_cols_raw

    # ② 正規化で列名衝突が起きた場合の去重は不要（ここではテンプレ順の1本化が完了）

    df = _sanitize_values_for_endpoint(df)

    if DEBUG_PAYLOAD_PREVIEW:
        print(">>> DEBUG dtypes(after sanitize):", {c: str(t) for c, t in df.dtypes.items()})
        if DEBUG_SHOW_SAMPLE_COL in normalize_columns(df.columns) and len(df) > 0:
            # デバッグ列名は正規化して比較、抽出は生名で
            # 生→正規化の対応
            raw_by_norm = {n: r for r, n in zip(df.columns, normalize_columns(df.columns))}
            raw = raw_by_norm.get(normalize_columns([DEBUG_SHOW_SAMPLE_COL])[0], None)
            if raw:
                print(f">>> DEBUG sample(after sanitize) '{DEBUG_SHOW_SAMPLE_COL}' =", repr(df[raw].iloc[0]))

    if cfg.limit_rows and len(df) > cfg.limit_rows:
        df = df.head(cfg.limit_rows).reset_index(drop=True)

    # バッチ分割
    bs = max(1, cfg.batch_size)
    n = len(df)
    if n == 0:
        return df.copy(), {"raw_responses": [], "notes": "empty input"}

    # 列整形（1回でよい）
    shaped = df.copy()

    bearer = _get_bearer(cfg.auth_mode, cfg.api_key)

    preds: List[Any] = []
    raw_responses: List[Any] = []
    print(f">>> AML scoring start: rows={n}, batch_size={bs}, endpoint={cfg.scoring_uri}", flush=True)
    debug_isolate = os.getenv("AML_DEBUG_ROW_ISOLATION", "0") in ("1", "true", "True")
    for i in range(0, n, bs):
        part = shaped.iloc[i:i+bs].reset_index(drop=True)
        if DEBUG_PAYLOAD_PREVIEW:
            print(">>> DEBUG: columns (sending) =", part.columns.tolist()[:30], "...")
            # 代表1行をプレビュー（ズレ検知）
            if len(part) > 0:
                preview = {c: part.iloc[0][c] for c in part.columns[:min(30, len(part.columns))]}
                print(">>> DEBUG: preview row (first ~30 cols) =", preview)

        # ★_payload_from_df を使わず、“テンプレの生columns＋同順data”で構築
        payload = _payload_from_df(part)
        
        print(f">>> posting batch {i}-{min(i+bs, n)-1} ({len(part)}) ...", flush=True)
        r = _post(cfg.scoring_uri, bearer, payload, cfg.connect_timeout_sec, cfg.timeout_sec, cfg.deployment)
        if r.status_code >= 400:
             # デプロイ名が不正なら一度だけヘッダ無しで再試行
             if r.status_code == 404 and cfg.deployment:
                 try_txt = (r.text or "").lower()
                 if "deployment could not be found" in try_txt:
                    r = _post(cfg.scoring_uri, bearer, payload, cfg.connect_timeout_sec, cfg.timeout_sec, deployment=None)
        if r.status_code >= 400:
            if debug_isolate and len(payload["input_data"]["data"]) > 1:
                for j, row in enumerate(payload["input_data"]["data"], start=0):
                    one = {"input_data": {"columns": payload["input_data"]["columns"], "data": [row]}}
                    r1 = _post(cfg.scoring_uri, bearer, one, cfg.connect_timeout_sec, cfg.timeout_sec, cfg.deployment)
                    if r1.status_code >= 400:
                        start_idx = i + j
                        raise RuntimeError(
                            f"HTTP {r1.status_code} at row {start_idx}: {r1.text}\n"
                            f"Failing row preview: {json.dumps({'input_data': {'columns': one['input_data']['columns'], 'data': [one['input_data']['data'][0]]}}, ensure_ascii=False)[:1500]}"
                        )     
            preview = {"input_data": {"columns": payload["input_data"]["columns"], "rows": len(payload['input_data']['data'])}}
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}\n\nPayload preview: {json.dumps(preview, ensure_ascii=False)}")

        # --- ここから応答のJSONパース（★欠けていた部分）---
        try:
            js = r.json()
        except Exception:
            raise RuntimeError(f"JSONデコードに失敗: {r.text[:500]}")
        raw_responses.append(js)

        # 代表的な返却形に対応
        # - [label, ...]  ← トップレベル配列
        # - {"result": [...]}, {"predictions": [...]}, {"output": [...]}
        # - その他の dict で最初の値が配列
        out = None
        if isinstance(js, list):
            out = js
        elif isinstance(js, dict):
            for key in ("result", "predictions", "output", "Results", "outputs", "values", "y_pred"):
                if key in js:
                    out = js[key]
                    break
            if out is None and js:
                first_val = next(iter(js.values()))
                if isinstance(first_val, (list, tuple)):
                    out = first_val
        if out is None:
            kind = type(js).__name__
            hint = f"list(len={len(js)})" if isinstance(js, list) else (f"dict(keys={list(js.keys())})" if isinstance(js, dict) else kind)
            raise RuntimeError(f"推論応答の形式を解釈できませんでした: type={kind}, hint={hint}")

        # 配列の中身がスカラーならそのまま、辞書/配列ならオブジェクトとして格納
        if isinstance(out, list) and len(out) and isinstance(out[0], (dict, list, tuple)):
            preds.extend(out)
        elif isinstance(out, list):
            preds.extend(out)
        else:
            preds.extend([out])
        print(">>> batch OK", flush=True)

    if len(preds) != n:
        raise RuntimeError(f"返却件数が一致しません (got={len(preds)}, expected={n})")

    result = df.copy()
    result[cfg.prediction_col] = preds
    return result, {"raw_responses": raw_responses, "expected_cols": expected_cols_norm}