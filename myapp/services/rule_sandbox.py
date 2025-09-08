# rule_sandbox.py （禁止構文チェックまわりを差し替え）
import ast

FORBIDDEN_TOKENS = {
    "keywords": ["if", "for", "while", "with", "try", "class", "def", "lambda", "global", "nonlocal"],
    "calls":    ["eval", "exec", "__import__"],
    "imports":  ["import", "from"],
    "literals": ["True", "False", "None"],  # ← True/False を禁止している設計ならここで捕捉
}

READABLE_HINTS = {
    "if":      "if 文は禁止です。条件分岐は pandas のブール条件（cond）を作って prediction.loc[cond] を使ってください。",
    "for":     "for 文は禁止です。反復は使わず、ベクトル化した条件式で書いてください。",
    "while":   "while 文は禁止です。",
    "with":    "with 文は禁止です。",
    "try":     "try/except は禁止です。",
    "class":   "class 定義は禁止です。",
    "def":     "def 定義は禁止です。apply_rules の外で関数は作れません。",
    "lambda":  "lambda は禁止です。",
    "global":  "global は禁止です。",
    "nonlocal":"nonlocal は禁止です。",
    "eval":    "eval() は禁止です。",
    "exec":    "exec() は禁止です。",
    "__import__": "__import__ は禁止です。",
    "import":  "import は禁止です。",
    "from":    "from import は禁止です。",
    "True":    "True は禁止です。代わりに 1 や 'true' 等で判定してください（例：pd.to_numeric(col, errors='coerce')==1）。",
    "False":   "False は禁止です。代わりに 0 や 'false' 等で判定してください。",
    "None":    "None は禁止です。欠損は df['col'].isna() で判定してください。",
}

def lint_forbidden(code: str):
    tree = ast.parse(code)
    found = []

    for node in ast.walk(tree):
        # import系
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append("import" if isinstance(node, ast.Import) else "from")
        # def/class
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append("def")
        if isinstance(node, ast.ClassDef):
            found.append("class")
        # 制御構文
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            name = node.__class__.__name__.lower()  # if/for/while/with/try
            found.append(name)
        # lambda / global / nonlocal
        if isinstance(node, ast.Lambda):
            found.append("lambda")
        if isinstance(node, ast.Global):
            found.append("global")
        if isinstance(node, ast.Nonlocal):
            found.append("nonlocal")
        # 危険呼び出し
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_TOKENS["calls"]:
                found.append(node.func.id)
        # True / False / None リテラル（Python 3.8+ は Constant）
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_TOKENS["literals"]:
            found.append(node.id)
        if isinstance(node, ast.Constant):
            if node.value is True:  found.append("True")
            if node.value is False: found.append("False")
            if node.value is None:  found.append("None")

    # 重複除去
    found = list(dict.fromkeys(found))

    if found:
        hints = [READABLE_HINTS.get(tok, f"'{tok}' は禁止です。") for tok in found]
        # ここで「何がダメだったか」を 1 行でまとめる
        human = "禁止構文を検出しました: " + ", ".join(found) + "。"
        advice = "／".join(hints)
        raise ValueError(f"{human}\n{advice}")
