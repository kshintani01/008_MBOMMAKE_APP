# myapp/services/aml_endpoint.py
import os, json, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests

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
    batch_size: int = int(os.getenv("AML_BATCH_SIZE", "256"))
    deployment: Optional[str] = os.getenv("AML_DEPLOYMENT")  # optional header
    prediction_col: str = os.getenv("ML_PREDICTION_COL", "prediction")

    @staticmethod
    def from_env() -> "AMLConfig":
        scoring_uri = os.getenv("AML_SCORING_URI")
        template = os.getenv("TEMPLATE_JSON")
        if not scoring_uri:
            raise RuntimeError("AML_SCORING_URI が未設定です。")
        if not template or not os.path.exists(template):
            raise RuntimeError("テンプレJSONが見つかりません。.env に TEMPLATE_JSON=./template.json を設定し、Studio Test タブの JSON を保存してください。")
        return AMLConfig(scoring_uri=scoring_uri, template_json_path=template)


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


def _post(endpoint: str, bearer: str, payload: Dict[str, Any], timeout: int, deployment: Optional[str]) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer}",
    }
    if deployment:
        headers["azureml-model-deployment"] = deployment
    return requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=timeout)


# ====== メイン: エンドポイント推論 ======
def score_via_endpoint(df: pd.DataFrame, cfg: AMLConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tpl, expected_cols = _load_template(cfg.template_json_path)

    # バッチ分割
    bs = max(1, cfg.batch_size)
    n = len(df)
    batches = []
    if n == 0:
        return df.copy(), {"raw_responses": [], "notes": "empty input"}

    # 列整形（1回でよい）
    shaped = _shape_df_for_template(df.copy(), expected_cols, cfg.drop_columns)

    bearer = _get_bearer(cfg.auth_mode, cfg.api_key)

    preds: List[Any] = []
    raw_responses: List[Any] = []

    for i in range(0, n, bs):
        part = shaped.iloc[i:i+bs].reset_index(drop=True)
        payload = _payload_from_df(part)
        r = _post(cfg.scoring_uri, bearer, payload, cfg.timeout_sec, cfg.deployment)
        if r.status_code >= 400:
            # 送信プレビューも返す
            preview = {"input_data": {"columns": payload["input_data"]["columns"], "rows": len(payload["input_data"]["data"])}}
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}\n\nPayload preview: {json.dumps(preview, ensure_ascii=False)}")
        try:
            js = r.json()
        except Exception:
            raise RuntimeError(f"JSONデコードに失敗: {r.text[:500]}")
        raw_responses.append(js)

        # 代表的な返却形に対応（AutoML/自作スコアリング）
        # - {"result": [label, ...]}
        # - {"predictions": [label, ...]}
        # - {"output": [label, ...]}
        out = None
        for key in ("result", "predictions", "output"):
            if key in js:
                out = js[key]
                break
        if out is None:
            # もし第一要素に更に入っていれば抽出を試みる
            if isinstance(js, dict) and js:
                first_val = next(iter(js.values()))
                if isinstance(first_val, list):
                    out = first_val
        if out is None:
            # 何が返ったか見えるようにメタへ格納して中断
            raise RuntimeError(f"推論応答の形式を解釈できませんでした: keys={list(js.keys())}")

        # 1次元ラベル配列を想定。スコア等の辞書ならそのまま詰める
        if isinstance(out, list) and len(out) and isinstance(out[0], (dict, list)):
            preds.extend(out)
        else:
            preds.extend(list(out))

    if len(preds) != n:
        raise RuntimeError(f"返却件数が一致しません (got={len(preds)}, expected={n})")

    result = df.copy()
    result[cfg.prediction_col] = preds
    return result, {"raw_responses": raw_responses, "expected_cols": expected_cols}