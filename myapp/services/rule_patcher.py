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
    before = base_code[:start]
    after  = base_code[end:]

    begin_line = m.group("prefix")
    # ★ BEGIN 行の直後に必ず改行を入れる
    if not begin_line.endswith("\n"):
        begin_line = begin_line + "\n"
    # normalized は末尾改行を1つ保証しているので END はそのまま連結でOK
    end_line = m.group("suffix")

    new_block = begin_line + indented_body + end_line
    return before + new_block + after

def unified_diff(old: str, new: str,
                 fromfile: str = "a",
                 tofile: str = "b",
                 context: int = 3) -> str:
    """
    ユニファイド差分を返す。context で前後コンテキスト行数を制御（0なら変更行のみ）。
    ルール本文は正規化（改行/インデント/末尾改行調整）してから比較。
    """
    old_norm = _normalize_rules_body(old)
    new_norm = _normalize_rules_body(new)
    diff = difflib.unified_diff(
        old_norm.splitlines(),          # keepends=False
        new_norm.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
        n=context
    )
    return "\n".join(diff)

def merge_rules_body_dedup(existing: str, addition: str) -> str:
    """
    既存RULES本文に追加分をマージ。行単位で重複は除外し、順序は保つ。
    空行・末尾空白は正規化してから比較。
    """
    def norm_lines(s: str) -> list[str]:
        lines = []
        for ln in (s or "").splitlines():
            t = ln.rstrip()
            if t.strip() == "":
                continue
            lines.append(t)
        return lines

    ex  = norm_lines(existing)
    add = norm_lines(addition)

    seen = set(ex)
    for ln in add:
        if ln not in seen:
            ex.append(ln)
            seen.add(ln)
    return "\n".join(ex)