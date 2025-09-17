import os
import re
from openai import AzureOpenAI
import pandas as pd
from azure.storage.blob import BlobServiceClient
from .rules_repo import extract_rules_body

# ===== 設定（環境変数） =====
ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT")

BLOB_CONN_STR   = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER  = os.getenv("AZURE_STORAGE_CONTAINER")
BLOB_BLOB_NAME  = os.getenv("AZURE_STORAGE_DEFAULT_CSV")

_client = AzureOpenAI(api_key=API_KEY, azure_endpoint=ENDPOINT, api_version=API_VERSION)

BEGIN_MASK = r"# === RULES:BEGIN ==="
END_MASK = r"# === RULES:END ==="

def _has_keys() -> bool:
    return bool(ENDPOINT and API_KEY and API_VERSION and DEPLOYMENT)

def choose_rules_body(cand_from_llm: str | None, fallback_default: str = "prediction[:] = '未分類'") -> str:
    """
    優先度: LLM生成候補 > デフォルト
    """
    cand = (cand_from_llm or "").strip()
    return cand

# ===== 便利関数 =====
def _looks_empty_condition(body: str) -> bool:
    """df.index.isin([]) や .isin([]) のような空集合条件を検出"""
    patterns = [
        r"\.index\.isin\(\s*\[\s*\]\s*\)",   # df.index.isin([])
        r"\.isin\(\s*\[\s*\]\s*\)",          # col.isin([])
    ]
    return any(re.search(p, body) for p in patterns)

def _references_any_df_column(body: str) -> bool:
    """df の列参照が1つも無い（df['...'] / df. ...）なら False"""
    return "df[" in body or "df." in body

def _format_schema_for_prompt(df):
    """プロンプトに埋め込む簡易スキーマ（列名・dtype・少数サンプル）"""
    if df is None:
        return "（列スキーマ未提供。列名を新規に発明しないこと）"
    lines = []
    for col in list(df.columns)[:40]:
        dtype = str(df[col].dtype)
        try:
            sample_vals = df[col].dropna().unique()[:5]
        except Exception:
            sample_vals = []
        sample_vals = ", ".join(map(lambda x: str(x)[:20], sample_vals))
        lines.append(f"- {col} (dtype={dtype}; samples=[{sample_vals}])")
    return "利用可能な列と型:\n" + "\n".join(lines)

def _strip_code_block(text: str) -> str:
    """```python ... ``` を剥がす（python ブロック優先→最長ブロック）"""
    t = (text or "").strip()
    if "```" not in t:
        return t
    parts = [p.strip() for p in t.split("```") if p.strip()]
    if not parts:
        return ""
    prefer = [p for p in parts if p.lower().startswith("python")]
    block = prefer[0] if prefer else max(parts, key=len)
    return block.replace("python", "", 1).strip()

def _llm_call_system_user(system_text: str, user_text: str) -> str:
    if not _has_keys():
        return ""

    resp = _client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

# ===== 参考：ダミーコード =====
CODE_EXAMPLE = (
    "import pandas as pd\nimport numpy as np\n\n"
    "def apply_rules(df):\n"
    "    prediction = pd.Series([0]*len(df), name='prediction')\n"
    "    return prediction\n"
)

_FORBIDDEN_PATTERNS = [
    r"prediction\s*\[\s*:\s*\]\s*=",   # prediction[:] = ...
    r"^\s*prediction\s*=",             # prediction = ...（再束縛）
    r"^\s*def\s+\w+\s*\(",             # ← 追加: 関数定義
    r"^\s*class\s+\w+\s*:",            # ← 追加: クラス定義
    r"^\s*import\s+\w+",               # ← 追加: import
    r"^\s*from\s+\w+\s+import\s+",     # ← 追加: from ... import ...
]

def _violates_forbidden(body: str) -> bool:
    t = body or ""
    return any(re.search(p, t, flags=re.MULTILINE) for p in _FORBIDDEN_PATTERNS)

def sanitize_rules_body(body: str) -> str:
    """禁止パターンがある場合は空にして再試行/フォールバックさせる"""
    body = (body or "").strip()
    if _violates_forbidden(body):
        return ""
    return body

def build_system_prompt():
    return ( "You are a Python data wrangler. Given Japanese natural language rules, " 
            "you will output ONLY Python code that defines a function:\n\n" 
            "def apply_rules(df):\n" 
            " # ...\n" 
            " return prediction\n" 
            )

def generate_code(natural_language: str) -> str:
    if not _has_keys():
        return CODE_EXAMPLE
    user_prompt = (
        "以下の自然言語ルールを満たす apply_rules(df) を返す Python コードだけを出力:\n\n"
        f"[ルール]\n{natural_language}\n"
    )
    content = _llm_call_system_user(build_system_prompt(), user_prompt).strip()
    content = _strip_code_block(content)
    return content or CODE_EXAMPLE

def healthcheck() -> str:
    if not _has_keys():
        return "ok"
    text = _llm_call_system_user("", "Reply with 'ok' only.").strip()
    return text or "ok"

def _load_default_df_from_blob() -> pd.DataFrame:
    """Blob Storage からデフォルト CSV を読み込む"""
    if not (BLOB_CONN_STR and BLOB_CONTAINER and BLOB_BLOB_NAME):
        raise RuntimeError("Blob Storage の設定が不足しています")
    service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    blob_client = service.get_blob_client(container=BLOB_CONTAINER, blob=BLOB_BLOB_NAME)
    stream = blob_client.download_blob()
    return pd.read_csv(stream)  # 文字コード・区切りは適宜調整

def _merge_rules_body(existing: str, addition: str) -> str:
    existing = (existing or "").rstrip()
    addition = (addition or "").rstrip()

    if not addition:
        return existing or "prediction[:] = '未分類'"
    if not existing:
        return addition
    return existing + "\n\n" + addition

# ===== 新：RULES ブロックの「中身だけ」を生成するモード =====
def generate_rules_body(natural_language: str, base_code: str, df=None) -> str:
    """
    自然言語ルールとベースコードから、
    # === RULES:BEGIN === と # === RULES:END === の間に入れる「中身だけ」を返す。
    df: DataFrame（あれば列スキーマをプロンプトに埋め込む）
    """
    existing_body = extract_rules_body(base_code) or ""

    if df is None:
        try:
            df = _load_default_df_from_blob()
        except Exception as e:
            print(f"[WARN] failed to load default CSV from Blob: {e}")        

    if not _has_keys():
        return ""

    sys_prompt = (
        "You are a Python data wrangler. Given Japanese natural language rules, "
        "output ONLY the body lines to insert between '# === RULES:BEGIN ===' and '# === RULES:END ===' "
        "inside apply_rules(df).\n\n"
        "Constraints:\n"
        "- Do NOT define functions/classes/imports/loops; use vectorized pandas only.\n"
        "- Allowed globals: pd, np, XREF1, XREF2 (read-only).\n"
        "- prediction is a pandas Series initialized with NaN. You MUST only write labels to rows that are still unset.\n"
        "- NEVER write lines that assign to the whole series (e.g., 'prediction[:] = ...') or rebind it (e.g., 'prediction = ...').\n"
        "- ALWAYS use this pattern: prediction.loc[(COND) & prediction.isna()] = 'ラベル'\n\n"
        "XREF usage hints:\n"
        "- mapping example: m = XREF2.lookup_map('key', 'label'); v = df['key'].map(m)\n"
        "- join example: _t = df.merge(XREF1.df[['part_no','family']], on='part_no', how='left')\n"
    )

    few_shot = (
        "# 既存には触れず、新しい条件だけを追加。未確定行にだけ付与して上書き防止。\n"
        "cond = df['引渡'].astype(str).str.strip().eq('G112')\n"
        "prediction.loc[cond & prediction.isna()] = 'WT137'\n"
    )

    schema_text = _format_schema_for_prompt(df)

    user_prompt = (
        "次の既存RULES本文を一切変更せず、新しい自然言語ルールを満たす『追加行のみ』を出力してください。\n"
        "既存本文:\n"
        f"-----EXISTING START-----\n{existing_body}\n-----EXISTING END-----\n\n"
        "注意:\n"
        " - 既存行の再出力・変更・削除は厳禁です。\n"
        " - 常に prediction.isna() による未確定チェックを併用し、既存付与を上書きしないでください。\n"
        " - 'prediction[:] = ...' や 'prediction = ...' のような全件代入は禁止です。\n\n"
        f"{schema_text}\n\n"
        f"{few_shot}\n"
        f"[自然言語ルール]\n{natural_language}\n"
    )

    # --- 初回生成 ---
    raw = _llm_call_system_user(sys_prompt, user_prompt).strip()
    body = _strip_code_block(raw)

    # --- 生成検査 → リトライ（空集合条件 or 列参照なし） ---
    need_retry = (not _references_any_df_column(body)) or _looks_empty_condition(body)
    if need_retry:
        reinforce = (
            "前回の出力は不適切でした。既存本文を変更せず、追加行だけを出力してください。"
            "必ずスキーマ内の列を参照し、空集合条件は使用しないでください。"
        )

        raw2 = _llm_call_system_user(sys_prompt, reinforce + "\n\n" + user_prompt).strip()
        body2 = _strip_code_block(raw2)
        if body2 and not _looks_empty_condition(body2) and _references_any_df_column(body2):
            body = body2

    body = sanitize_rules_body(body)

    # --- 空 or pass を物理的に回避 ---
    if not body or body.strip() in {"pass", "# 生成結果なし"}:
        body = "# no-op (fallback): 未確定は full.py の fillna('未分類') で埋められます"

    return body
