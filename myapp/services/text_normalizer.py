import unicodedata
import pandas as pd
import re

CANONICAL_COLUMN_ALIASES = {
    "ﾕﾆｯﾄID": "ユニットID",
    "ﾕﾆｯﾄId": "ユニットID",
    "ﾕﾆｯﾄｉｄ": "ユニットID",
    # 必要に応じて追加
}

# 制御・不可視文字（ゼロ幅等）を除去
_CONTROL_INVIS_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")  # ZWSP/ZWNJ/ZWJ/BOM
_NBSP = "\u00A0"

def _clean_str(s: str) -> str:
    # NFKC → 不可視除去 → NBSP→通常スペース → 前後strip
    s = unicodedata.normalize("NFKC", s)
    s = _CONTROL_INVIS_PATTERN.sub("", s).replace(_NBSP, " ")
    s = s.strip()
    return s

def nkfc(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return s
    return _clean_str(str(s))

def normalize_columns(cols):
    out = []
    for c in cols:
        v = nkfc(c)
        v = CANONICAL_COLUMN_ALIASES.get(v, v)
        out.append(v)
    return out

def normalize_df_kana(df: pd.DataFrame) -> pd.DataFrame:
    # 列名 正規化+別名
    new_cols = normalize_columns(df.columns)
    df = df.copy()
    df.columns = new_cols

    # 文字列セル 正規化（object / string）
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].map(nkfc)

    # 列名衝突を去重（左勝ち、非NA優先マージ）
    if df.columns.duplicated().any():
        merged = {}
        for col in df.columns:
            if col not in merged:
                merged[col] = df[col]
            else:
                s = merged[col].copy()
                mask = s.isna()
                s.loc[mask] = df[col].loc[mask]
                merged[col] = s
        df = pd.DataFrame(merged)

    return df