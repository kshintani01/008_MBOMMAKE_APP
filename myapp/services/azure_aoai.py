import os
from openai import AzureOpenAI

# ===== 設定（環境変数） =====
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# 鍵はログに出さない
# print(ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT)

_client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION,
)

def _model_name() -> str:
    # settings ではなく env（DEPLOYMENT）に統一
    return DEPLOYMENT


# ===== 参考：ダミーコード =====
CODE_EXAMPLE = (
    "import pandas as pd\nimport numpy as np\n\n"
    "def apply_rules(df):\n"
    "    prediction = pd.Series([0]*len(df), name='prediction')\n"
    "    return prediction\n"
)


# ===== 共通：Responses からテキスト抽出（空返し対策） =====
def _extract_text_from_responses(resp):
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


# ===== 旧: 関数ごと生成するモード（そのまま） =====
def build_system_prompt():
    return (
        "You are a Python data wrangler. Given Japanese natural language rules, "
        "you will output ONLY Python code that defines a function:\n\n"
        "def apply_rules(df):\n"
        "    # ...\n"
        "    return prediction\n"
    )


def generate_code(natural_language: str) -> str:
    # キー未設定ならダミー返す
    if not (ENDPOINT and API_KEY and API_VERSION and DEPLOYMENT):
        return CODE_EXAMPLE

    prompt = (
        "以下の自然言語ルールを満たす apply_rules(df) を返す Python コードだけを出力:\n\n"
        f"[ルール]\n{natural_language}\n"
    )
    resp = _client.responses.create(
        model=_model_name(),
        input=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
    )
    content = _extract_text_from_responses(resp).strip()

    # コードフェンスがあれば、python ブロック優先→最長ブロック
    if "```" in content:
        parts = [p.strip() for p in content.split("```") if p.strip()]
        prefer = [p for p in parts if p.lower().startswith("python")]
        block = (prefer[0] if prefer else max(parts, key=len))
        content = block.replace("python", "", 1).strip()

    return content or CODE_EXAMPLE


def healthcheck() -> str:
    if not (ENDPOINT and API_KEY and API_VERSION and DEPLOYMENT):
        return "ok"  # ローカル無設定時の仮OK
    prompt = "Reply with 'ok' only."
    r = _client.responses.create(model=_model_name(), input=prompt)
    text = _extract_text_from_responses(r).strip()
    return text or "ok"


# ===== 新: RULES ブロックの「中身だけ」を生成するモード =====
def generate_rules_body(natural_language: str, base_code: str) -> str:
    """
    自然言語ルールとベースコードから、
    # === RULES:BEGIN === と # === RULES:END === の間に入れる「中身だけ」を返す。
    """
    if not (ENDPOINT and API_KEY and API_VERSION and DEPLOYMENT):
        return "prediction[:] = '未分類'  # fallback(no-keys)"

    sys_prompt = (
        "You are a Python data-wrangling assistant.\n"
        "Task: Output ONLY the body lines to insert between '# === RULES:BEGIN ===' and '# === RULES:END ===' "
        "inside apply_rules(df).\n"
        "Hard requirements:\n"
        " - NEVER output 'pass'.\n"
        " - ALWAYS write at least one assignment to prediction using pandas boolean indexing, e.g.:\n"
        "     prediction.loc[cond] = 'ラベル'\n"
        " - If rules are ambiguous, FIRST set a default for all rows:\n"
        "     prediction[:] = '未分類'\n"
        " - Do not use imports, I/O, function/class defs, eval/exec, with/try, lambda, globals.\n"
        " - You may only use 'df' and 'prediction', and pandas/numpy ops already available.\n"
        " - Do NOT modify df's schema; write labels ONLY into 'prediction'.\n"
        " - Output executable Python statements only (no fences, no comments-only output).\n"
        "Interface assumptions:\n"
        " - 'prediction' is a pandas.Series aligned to df.index and will be returned by apply_rules.\n"
        "Quality bar:\n"
        " - Prefer concise boolean expressions with clear parentheses.\n"
        " - Normalize strings with .str.strip() when comparing.\n"
    )

    few_shot = (
        "【良い例1（日本語→コード）】\n"
        "自然言語: 「数量が10超かつ品目区分が'加工'なら '部品製作'、それ以外は'未分類'」\n"
        "出力ボディ:\n"
        "prediction[:] = '未分類'\n"
        "cond = (pd.to_numeric(df['数量'], errors='coerce') > 10) & (df['品目区分'].astype(str).str.strip() == '加工')\n"
        "prediction.loc[cond] = '部品製作'\n"
        "\n"
        "【良い例2】\n"
        "自然言語: 「材質が'鋼'で長さ>=1000、かつ数量>=2なら'外注製作'」\n"
        "出力ボディ:\n"
        "cond = (\n"
        "    df['材質'].astype(str).str.strip() == '鋼'\n"
        ") & (pd.to_numeric(df['長さ'], errors='coerce') >= 1000) & (pd.to_numeric(df['数量'], errors='coerce') >= 2)\n"
        "prediction.loc[cond] = '外注製作'\n"
    )

    user_prompt = (
        "ベースコードの RULES ブロックに挿入する『中身だけ』を、Python実行文のみで出力してください。\n"
        "必ず少なくとも1つは prediction.loc[...] = 'ラベル' の代入を行い、pass は絶対に出力しないでください。\n"
        "曖昧な場合は最初に prediction[:] = '未分類' としてください。\n\n"
        f"{few_shot}\n"
        "[自然言語ルール]\n{natural_language}\n\n"
        "[ベースコード]\n{base_code}\n"
    )

    resp = _client.responses.create(
        model=_model_name(),
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = _extract_text_from_responses(resp).strip()

    # フェンス剥がし（python ブロック優先→最長）
    body = raw
    if "```" in raw:
        parts = [p.strip() for p in raw.split("```") if p.strip()]
        prefer = [p for p in parts if p.lower().startswith("python")]
        body = (prefer[0] if prefer else max(parts, key=len))
        body = body.replace("python", "", 1).strip()

    # 空 or pass を物理的に回避
    if not body or body.strip() in {"pass", "# 生成結果なし"}:
        body = "prediction[:] = '未分類'  # fallback"

    # 最低1本の代入を保証（無ければダミー1行を追加）
    if "prediction.loc[" not in body:
        if "prediction[:]" not in body:
            body = "prediction[:] = '未分類'\n" + body
        body += "\n# ensure at least one assignment\nprediction.loc[pd.Series(False, index=df.index)] = '未分類'"

    return body
