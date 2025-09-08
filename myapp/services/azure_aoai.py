import os
import re
from openai import AzureOpenAI

# ===== 設定（環境変数） =====
ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT")

_client = AzureOpenAI(api_key=API_KEY, azure_endpoint=ENDPOINT, api_version=API_VERSION)

def _model_name() -> str:
    return DEPLOYMENT

def _has_keys() -> bool:
    return bool(ENDPOINT and API_KEY and API_VERSION and DEPLOYMENT)

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

# ===== 共通：テキスト抽出（Responses / Chat 両対応） =====
def _extract_text_from_responses(resp) -> str:
    # 新Responses API
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    try:
        for out in getattr(resp, "output", []) or []:
            for c in getattr(out, "content", []) or []:
                if getattr(c, "type", "") in {"output_text", "text"} and getattr(c, "text", ""):
                    return c.text
    except Exception:
        pass
    return ""

def _extract_text_from_chat(resp) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

# ===== 互換レイヤー：responses が無ければ chat.completions を使う =====
def _supports_responses() -> bool:
    return hasattr(_client, "responses")

def _llm_call_system_user(system_text: str, user_text: str) -> str:
    model = _model_name()
    if not _has_keys():
        return ""
    if _supports_responses():
        # Responses API（新）
        resp = _client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
        )
        return _extract_text_from_responses(resp).strip()
    else:
        # Chat Completions API（旧）
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
            temperature=0,
        )
        return _extract_text_from_chat(resp)

# ===== 参考：ダミーコード =====
CODE_EXAMPLE = (
    "import pandas as pd\nimport numpy as np\n\n"
    "def apply_rules(df):\n"
    "    prediction = pd.Series([0]*len(df), name='prediction')\n"
    "    return prediction\n"
)

# ===== 旧：関数ごと生成するモード =====
def build_system_prompt():
    return (
        "You are a Python data wrangler. Given Japanese natural language rules, "
        "you will output ONLY Python code that defines a function:\n\n"
        "def apply_rules(df):\n"
        "    # ...\n"
        "    return prediction\n"
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

# ===== 新：RULES ブロックの「中身だけ」を生成するモード =====
def generate_rules_body(natural_language: str, base_code: str, df=None) -> str:
    """
    自然言語ルールとベースコードから、
    # === RULES:BEGIN === と # === RULES:END === の間に入れる「中身だけ」を返す。
    df: DataFrame（あれば列スキーマをプロンプトに埋め込む）
    """
    if not _has_keys():
        return "prediction[:] = '未分類'  # fallback(no-keys)"

    # --- system ---
    sys_prompt = (
        "You are a Python data-wrangling assistant.\n"
        "Task: Output ONLY the body lines to insert between '# === RULES:BEGIN ===' and '# === RULES:END ==='.\n"
        "Hard requirements:\n"
        " - NEVER output 'pass'.\n"
        " - ALWAYS write at least one assignment to prediction using pandas boolean indexing:\n"
        "     prediction.loc[cond] = 'ラベル'\n"
        " - If rules are ambiguous, FIRST set a default:\n"
        "     prediction[:] = '未分類'\n"
        " - You MUST reference at least ONE EXISTING COLUMN from the provided schema.\n"
        " - DO NOT invent column names.\n"
        " - FORBIDDEN: empty-set conditions such as df.index.isin([]) or .isin([]) with an empty list.\n"
        " - No imports / I/O / defs / eval/exec / with/try / lambda / globals.\n"
        " - Only use 'df' and 'prediction' (pandas/numpy already available).\n"
        " - Do NOT modify df's schema; assign labels ONLY into 'prediction'.\n"
        "Output executable Python statements only (no fences).\n"
        "Hints:\n"
        " - For booleans in Japanese rules like “XがTrue”, accept True/1/'true' (case-insensitive) safely.\n"
    )

    # --- few-shot ---
    few_shot = (
        "【良い例1（文字×数値）】\n"
        "prediction[:] = '未分類'\n"
        "cond = (pd.to_numeric(df['数量'], errors='coerce') > 10) & (df['品目区分'].astype(str).str.strip() == '加工')\n"
        "prediction.loc[cond] = '部品製作'\n"
        "\n"
        "【良い例2（ANDで数値×数値×文字）】\n"
        "cond = (\n"
        "    df['材質'].astype(str).str.strip() == '鋼'\n"
        ") & (pd.to_numeric(df['長さ'], errors='coerce') >= 1000) & (pd.to_numeric(df['数量'], errors='coerce') >= 2)\n"
        "prediction.loc[cond] = '外注製作'\n"
        "\n"
        "【良い例3（ブール列）】\n"
        "cond = (\n"
        "    (df['sample'] == True) |\n"
        "    (pd.to_numeric(df['sample'], errors='coerce') == 1) |\n"
        "    (df['sample'].astype(str).str.lower().str.strip().isin(['true','t','1','yes','y']))\n"
        ")\n"
        "prediction.loc[cond] = '製作A'\n"
    )

    schema_text = _format_schema_for_prompt(df)

    # --- user ---
    user_prompt = (
        "ベースコードの RULES ブロックに挿入する『中身だけ』を、Python実行文のみで出力してください。\n"
        "必ず少なくとも1つは prediction.loc[...] = 'ラベル' の代入を行い、pass は絶対に出力しないでください。\n"
        "曖昧な場合は最初に prediction[:] = '未分類' としてください。\n\n"
        f"{schema_text}\n\n"
        f"{few_shot}\n"
        "[自然言語ルール]\n{natural_language}\n\n"
        "[ベースコード]\n{base_code}\n"
    )

    # --- 初回生成 ---
    raw = _llm_call_system_user(sys_prompt, user_prompt).strip()
    body = _strip_code_block(raw)

    # --- 生成検査 → リトライ（空集合条件 or 列参照なし） ---
    need_retry = (not _references_any_df_column(body)) or _looks_empty_condition(body)
    if need_retry:
        reinforce = (
            "前回の出力では既存列の参照が無い、または空集合条件が含まれていました。"
            "必ずスキーマ内の既存列を少なくとも1つ参照し、空集合条件（df.index.isin([]) 等）は使用しないでください。"
        )
        raw2 = _llm_call_system_user(sys_prompt, reinforce + "\n\n" + user_prompt).strip()
        body2 = _strip_code_block(raw2)
        if body2 and not _looks_empty_condition(body2) and _references_any_df_column(body2):
            body = body2

    # --- 空 or pass を物理的に回避 ---
    if not body or body.strip() in {"pass", "# 生成結果なし"}:
        body = "prediction[:] = '未分類'  # fallback"

    # --- 最低1本の代入を保証 ---
    if "prediction.loc[" not in body:
        if "prediction[:]" not in body:
            body = "prediction[:] = '未分類'\n" + body
        body += "\n# ensure at least one assignment\nprediction.loc[pd.Series(False, index=df.index)] = '未分類'"

    return body
