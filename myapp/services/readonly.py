# myapp/services/readonly.py
import pandas as pd

class ReadOnlyDF:
    """
    最低限の参照操作に限定した DataFrame ラッパ。
    ミューテーション系は例外を投げる。
    """
    __slots__ = ("_df",)

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy(deep=False)  # 共有参照（軽量）

    # よく使う参照メソッドのみ露出
    def head(self, n=5): return self._df.head(n)
    def columns(self): return self._df.columns
    def to_dict(self, orient="list"): return self._df.to_dict(orient=orient)

    # 代表的な参照系
    def query(self, expr, **kw): return self._df.query(expr, **kw)
    def merge(self, right, **kw):  # right は通常 df 側
        # merge は df 側で実施する想定だが、利便性のため許可
        if isinstance(right, ReadOnlyDF):
            right = right._df
        return self._df.merge(right, **kw)

    # よく使う vlookup 的ユーティリティ
    def lookup_map(self, key_col: str, value_col: str):
        """Series: key -> value の辞書化（欠損はそのまま）"""
        return self._df.set_index(key_col)[value_col].to_dict()

    # ミューテーション阻止
    def __setattr__(self, name, value):
        if name == "_df":
            object.__setattr__(self, name, value)
            return
        raise AttributeError("ReadOnlyDF is immutable")

    def __setitem__(self, *args, **kwargs):
        raise TypeError("ReadOnlyDF does not allow item assignment")

    @property
    def df(self):
        """本体 DF が必要なときのみ（読み取り想定）"""
        return self._df.copy(deep=False)
