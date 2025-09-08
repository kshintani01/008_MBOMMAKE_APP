# services/rule_patcher.py
import difflib
import re
import textwrap

MARK_BEGIN = r"# === RULES:BEGIN ==="
MARK_END   = r"# === RULES:END ==="

_BEGIN_RE = re.compile(rf"^(?P<indent>[ \t]*){re.escape(MARK_BEGIN)}\s*$", re.MULTILINE)
_BLOCK_RE = re.compile(
    rf"(?P<prefix>^(?P<indent>[ \t]*){re.escape(MARK_BEGIN)}\s*$)"
    r"(?P<body>.*?)"
    rf"(?P<suffix>^[ \t]*{re.escape(MARK_END)}\s*$)",
    re.DOTALL | re.MULTILINE
)

def _normalize_rules_body(body: str) -> str:
    """
    - 文字コード周り・改行を統一
    - 先頭/末尾の空行を除去
    - 共通インデントをデデント
    - タブはスペースに
    """
    if body is None:
        return ""
    s = body.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", "    ")
    s = s.strip("\n")
    s = textwrap.dedent(s)
    # 末尾に必ず改行を1つ
    if not s.endswith("\n"):
        s += "\n"
    return s

def replace_rules_block(base_code: str, new_rules_body: str) -> str:
    """
    base_code 内の RULES ブロックを new_rules_body で置換して返す。
    new_rules_body はマーカーを含まない「中身のみ」。
    """
    m = _BLOCK_RE.search(base_code)
    if not m:
        raise ValueError("RULES ブロック（BEGIN/END）がベースコードに見つかりません。")
    
    base_indent = m.group("indent")
    normalized = _normalize_rules_body(new_rules_body)

    indented_body_lines = []
    for line in normalized.splitlines(True):
        if line.strip():
            indented_body_lines.append(base_indent + line)
        else:
            indented_body_lines.append(line)
    indented_body = "".join(indented_body_lines)

    start, end = m.span()
    prefix = base_code[:start]
    suffix = base_code[end:]

    new_block = m.group("prefix") + indented_body + m.group("suffix")

    return prefix + new_block + suffix

def unified_diff(old: str, new: str, fromfile="base.py", tofile="new.py") -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(True), new.splitlines(True),
            fromfile=fromfile, tofile=tofile, lineterm=""
        )
    )
