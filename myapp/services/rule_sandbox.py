# myapp/services/rule_sandbox.py
import ast
from types import MappingProxyType
from typing import Any, Dict

import pandas as pd
import numpy as np


class SandboxError(Exception):
    """安全実行に失敗したときに投げるアプリ固有の例外。"""
    pass


# ---- 禁止構文の検査（安全のため最小限に） ---------------------------------

READABLE_HINTS = {
    "import": "import は禁止です。外部モジュールの読み込みは不可です。",
    "from": "from ... import は禁止です。",
    "exec": "exec() は禁止です。",
    "eval": "eval() は禁止です。",
    "def": "def 定義は禁止です。関数定義はできません。",
    "class": "class 定義は禁止です。",
    "with": "with 文は禁止です。",
    "try": "try/except は禁止です。",
    "lambda": "lambda は禁止です。",
    "global": "global は禁止です。",
    "nonlocal": "nonlocal は禁止です。",
    # if/for/while を完全禁止にすると記述力が落ちるためデフォルトは許可。
    # pandas のベクトル化スタイルを推奨したい場合は下をアンコメント。
    # "if": "if 文は禁止です。条件は cond を作り prediction.loc[cond] = 'ラベル' としてください。",
    # "for": "for 文は禁止です。ベクトル化された条件で書いてください。",
    # "while": "while 文は禁止です。ループは使用できません。",
}

FORBIDDEN_CALLS = {"exec", "eval", "__import__"}

def _lint_forbidden(code: str) -> None:
    """
    危険なノードを検出して SandboxError を投げる。
    True/False/None は許可（以前は禁止でしたが実用性のため解禁）。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxError(f"構文エラー: {e}")

    found = []

    for node in ast.walk(tree):
        # import 系
        if isinstance(node, ast.Import):
            found.append("import")
        elif isinstance(node, ast.ImportFrom):
            found.append("from")

        # def/class
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append("def")
        elif isinstance(node, ast.ClassDef):
            found.append("class")

        # 制御構文（必要があれば禁止に）
        # elif isinstance(node, ast.If):
        #     found.append("if")
        # elif isinstance(node, ast.For):
        #     found.append("for")
        # elif isinstance(node, ast.While):
        #     found.append("while")
        elif isinstance(node, ast.With):
            found.append("with")
        elif isinstance(node, ast.Try):
            found.append("try")

        # その他危険
        elif isinstance(node, ast.Lambda):
            found.append("lambda")
        elif isinstance(node, ast.Global):
            found.append("global")
        elif isinstance(node, ast.Nonlocal):
            found.append("nonlocal")

        # 危険な呼び出し
        elif isinstance(node, ast.Call):
            # 名前呼び出し exec/eval/__import__
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                found.append(node.func.id)

    if found:
        # 重複除去
        uniq = list(dict.fromkeys(found))
        human = "禁止構文を検出しました: " + ", ".join(uniq) + "。"
        advice = "／".join(READABLE_HINTS.get(tok, f"'{tok}' は禁止です。") for tok in uniq)
        raise SandboxError(f"{human}\n{advice}")


# ---- 実行系（強く制限した環境で exec → apply_rules 呼び出し） ---------------

_ALLOWED_BUILTINS: Dict[str, Any] = {
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "all": all,
    "any": any,
    "enumerate": enumerate,
    "zip": zip,
    # True/False/None は Python のリテラルとして解釈されるので builtins には不要
}
_READONLY_BUILTINS = MappingProxyType(_ALLOWED_BUILTINS)


def _wrap_if_body_only(code: str) -> str:
    """
    受け取った code が apply_rules を含まない（＝RULES の“中身だけ”）場合に、
    安全な雛形でラップしてフル関数化する。
    """
    if "def apply_rules" in code:
        return code

    # RULES ボディだけを function に包む
    return (
        "import pandas as pd\n"
        "import numpy as np\n\n"
        "def apply_rules(df):\n"
        "    prediction = pd.Series(index=df.index, dtype=object, name='prediction')\n"
        "    # === RULES:BEGIN ===\n"
        + "".join("    " + line for line in code.splitlines(True))
        + "\n"
        "    # === RULES:END ===\n"
        "    return prediction\n"
    )


def run_on_dataframe(code: str, df: pd.DataFrame):
    """
    生成コードをサンドボックスで実行し、apply_rules(df) の戻り値（pd.Series）を返す。
    - 禁止構文チェック
    - 制限付き名前空間で exec
    - 戻り値検証（Series / 長さ一致 / name='prediction' を推奨）
    """
    if not isinstance(code, str) or not code.strip():
        raise SandboxError("実行コードが空です。生成に失敗している可能性があります。")

    # 1) リンター（禁止構文の検査）
    _lint_forbidden(code)

    # 2) apply_rules を含まない場合は雛形でラップ
    source = _wrap_if_body_only(code)

    # 3) 実行用の名前空間（超制限）
    safe_globals = {
        "__builtins__": _READONLY_BUILTINS,
        "pd": pd,
        "np": np,
    }
    safe_locals: Dict[str, Any] = {}

    # 4) exec（コンパイルは別段階で）
    try:
        compiled = compile(source, filename="<sandbox>", mode="exec", dont_inherit=True)
        exec(compiled, safe_globals, safe_locals)
    except SandboxError:
        raise
    except Exception as e:
        raise SandboxError(f"コード実行エラー: {e}")

    # 5) 関数取り出し
    fn = safe_locals.get("apply_rules") or safe_globals.get("apply_rules")
    if not callable(fn):
        raise SandboxError("apply_rules(df) が見つかりません。コード生成に失敗しています。")

    # 6) 関数実行
    try:
        result = fn(df)
    except Exception as e:
        raise SandboxError(f"apply_rules(df) 実行時エラー: {e}")

    # 7) 戻り値検証
    if not isinstance(result, pd.Series):
        raise SandboxError("apply_rules(df) の戻り値が pandas.Series ではありません。")
    if len(result) != len(df):
        raise SandboxError(f"戻り値の長さが一致しません（期待: {len(df)}, 実際: {len(result)}）。")
    # name が未設定なら付ける
    if result.name is None:
        result = result.rename("prediction")

    return result
