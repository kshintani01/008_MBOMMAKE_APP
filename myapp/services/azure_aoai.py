import os
# services/azure_aoai.py
from django.conf import settings
from openai import AzureOpenAI

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

_client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION,
)

_MODEL = settings.AZURE_OPENAI_DEPLOYMENT

def _model_name() -> str:
    return DEPLOYMENT

def build_system_prompt():
    return (
        "You are a Python data wrangler. Given Japanese natural language rules, "
        "you will output ONLY Python code that defines a function:\n\n"
        "def apply_rules(df):\n"
        "    # ...\n"
        "    return prediction\n"
    )

CODE_EXAMPLE = (
    "import pandas as pd\nimport numpy as np\n\n"
    "def apply_rules(df):\n"
    "    prediction = pd.Series([0]*len(df), name='prediction')\n"
    "    return prediction\n"
)


def generate_code(natural_language: str) -> str:
    """キー未設定ならダミーコードを返す"""
    if not _client:
        return CODE_EXAMPLE

    prompt = (
        "以下の自然言語ルールを満たす apply_rules(df) を返す Python コードだけを出力:\n\n"
        f"[ルール]\n{natural_language}\n"
    )
    resp = _client.responses.create(
        model=DEPLOYMENT,
        input=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.output_text
    if "```" in content:
        content = content.split("```", 2)
        if len(content) == 3:
            content = content[1]
        content = content.replace("python\n", "")
    return content.strip() or CODE_EXAMPLE

def healthcheck() -> str:
    prompt = "Reply with 'ok' only."
    if hasattr(_client, "responses"):
        r = _client.responses.create(model=_MODEL, input=prompt)
        text = getattr(r, "output_text", None)
        if not text:
            # 2) 新Responses APIの生構造
            try:
                text = r.choices[0].message.content
            except Exception:
                text = ""
        return text or ""
    else:
        r = _client.chat.completions.create(model=_MODEL, messages=[{"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "").strip()

def generate_rules_body(natural_language: str, base_code: str) -> str:
    """
    自然言語ルールとベースコードを渡すと、
    # === RULES:BEGIN === と # === RULES:END === の間に入れる「中身だけ」を返す。
    """
    if not hasattr(_client, "responses"):  # フォールバック時はダミー
        return "# TODO: ここに条件を記述\npass"

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
        " - You may only use variables 'df' and 'prediction', and pandas/numpy operations already available.\n"
        " - Do NOT modify df's schema; write labels ONLY into 'prediction'.\n"
        " - The output must contain executable Python statements only (no fences, no comments-only output).\n"
        "Interface assumptions:\n"
        " - 'prediction' is a pandas.Series aligned to df.index and will be returned by apply_rules.\n"
        " - Target semantic column is 日本語の製作種別 labels (strings), but you only assign to prediction.\n"
        "Quality bar:\n"
        " - Prefer concise boolean expressions with clear precedence parentheses.\n"
        " - When comparing strings, normalize with .str.strip() if needed.\n"
    )

    few_shot = (
        "【良い例1（日本語→コード）】\n"
        "自然言語: 「数量が10超かつ品目区分が'加工'なら '部品製作'、それ以外は'未分類'」\n"
        "出力ボディ:\n"
        "prediction[:] = '未分類'\n"
        "cond = (df['数量'] > 10) & (df['品目区分'].astype(str).str.strip() == '加工')\n"
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
        model=DEPLOYMENT,
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    body = (getattr(resp, "output_text", "") or "").strip()
    # 万一コードフェンスが混ざってきたら剥がしておく
    if "```" in body:
        parts = [p.strip() for p in body.split("```") if p.strip()]
        body = parts[-1] if parts else ""

    # 最低限の安全補完（空 or pass 回避）
    if not body or body.strip() in {"pass", "# 生成結果なし"}:
        body = "prediction[:] = '未分類'  # fallback\n"

    return body or "# 生成結果なし"

def _extract_text_from_responses(resp):
    # 1) まず output_text
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    # 2) messages 構造を総なめ（Azure Responses の素の形式対策）
    try:
        for out in resp.output:
            for c in getattr(out, "content", []):
                if getattr(c, "type", "") == "output_text" and getattr(c, "text", ""):
                    return c.text
                if getattr(c, "type", "") == "text" and getattr(c, "text", ""):
                    return c.text
    except Exception:
        pass
    return ""