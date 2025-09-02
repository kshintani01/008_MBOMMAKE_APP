import types
import pandas as pd
import numpy as np
import builtins
import ast


ALLOWED_BUILTINS = {"range", "len", "min", "max", "sum", "abs", "all", "any", "enumerate"}
ALLOWED_GLOBALS = {"pd": pd, "np": np}


# 許可ノード（最低限）
_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Load, ast.Store,
    ast.Return, ast.Assign, ast.AugAssign, ast.Expr, ast.Call, ast.Name,
    ast.Attribute, ast.Subscript, ast.Slice, ast.BinOp, ast.BoolOp, ast.UnaryOp,
    ast.Compare, ast.If, ast.For, ast.While, ast.Constant, ast.List, ast.Dict,
    ast.Tuple, ast.ListComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
)


class SandboxError(Exception):
    pass


def _check_ast_tree(code_text: str):
    tree = ast.parse(code_text)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SandboxError(f"禁止構文を検出: {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "__import__":
                raise SandboxError("__import__は禁止")
    return tree




def exec_user_code(code_text: str):
    """コードを検証・実行し、apply_rules 関数を返す"""
    _check_ast_tree(code_text)
    safe_builtins = {k: getattr(builtins, k) for k in ALLOWED_BUILTINS if hasattr(builtins, k)}
    env = {"__builtins__": safe_builtins, **ALLOWED_GLOBALS}


    exec(compile(code_text, "<user_code>", "exec"), env, env)
    fn = env.get("apply_rules")
    if not isinstance(fn, types.FunctionType):
        raise SandboxError("apply_rules(df) 関数が見つかりません")
    return fn

def run_on_dataframe(code_text: str, df: pd.DataFrame) -> pd.Series:
    fn = exec_user_code(code_text)
    out = fn(df.copy())
    if not isinstance(out, pd.Series):
        raise SandboxError("apply_rules は pandas.Series を返す必要があります")
    if len(out) != len(df):
        raise SandboxError("返却 Series の長さが df と一致しません")
    if out.name is None:
        out.name = "prediction"
    return out