import os

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "dummy-model"

_client = None
if AzureOpenAI and ENDPOINT and API_KEY:
    _client = AzureOpenAI(
        api_key=API_KEY,
        api_version="2024-12-01-preview",
        azure_endpoint=ENDPOINT,
    )


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
        temperature=0.2,
        max_output_tokens=800,
    )
    content = resp.output_text
    if "```" in content:
        content = content.split("```", 2)
        if len(content) == 3:
            content = content[1]
        content = content.replace("python\n", "")
    return content.strip() or CODE_EXAMPLE
