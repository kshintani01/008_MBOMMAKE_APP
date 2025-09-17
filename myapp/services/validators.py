# services/validators.py（新規）
from django.conf import settings

def warn_unknown_columns(df):
    expected = set(getattr(settings, "RULES_COLUMNS", []))
    if not expected:
        return []
    actual = set(map(str, df.columns))
    # 実データにあるけど宣言に無い → 使われないなら問題ないが、一応情報表示
    extra = sorted(actual - expected)
    # 宣言してるけど今回のデータに無い → 使うとKeyErrorになる可能性あり
    missing = sorted(expected - actual)
    msgs = []
    if extra:
        msgs.append(f"CSVに想定外の列があります: {extra}")
    if missing:
        msgs.append(f"宣言済みだがCSVに存在しない列があります: {missing}")
    return msgs
