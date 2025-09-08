import os
# services/azure_aoai.py
from django.conf import settings
from openai import AzureOpenAI

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

print(ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT)

_client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION,
)

_MODEL = settings.AZURE_OPENAI_DEPLOYMENT


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
        "You are a Python data wrangler.\n"
        "Edit ONLY the rules area for apply_rules(df).\n"
        "Return ONLY the body (lines) to be placed between the markers,\n"
        "without code fences, without BEGIN/END markers.\n"
        "Target column is '製作種別' (use prediction Series to write labels).\n"
        "Allowed operations: pandas boolean filtering, assignments to prediction.\n"
        "STRICT PROHIBITONS: DO NOT use 'import' or 'from ... import',\n"
        "Use pandas boolean indexing like: prediction.loc[cond] = 'xxx'\n"
        "Do not modify variables other than 'prediction' and 'df'.\n"
        "no function/class definitions, no I/O, no eval/exec, no global/nonlocal,\n"
        "no external modules. Use existing 'df' and 'prediction' only.\n"
    )


    user_prompt = (
        "ベースコードの RULES ブロックに挿入する『中身だけ』を出力してください。\n"
        "コードフェンスやマーカーは出力しないでください。\n"
        "[自然言語ルール]\n"
        f"{natural_language}\n\n"
        "[ベースコード]\n"
        f"{base_code}\n"
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
        parts = body.split("```")
        body = parts[1] if len(parts) > 1 else body
        body = body.replace("python", "").strip()
    return body or "# 生成結果なし"