import pandas as pd
import numpy as np

def apply_rules(df):
    # df の行と 1:1 に対応する予測列（文字ラベル想定）
    prediction = pd.Series(index=df.index, dtype=object, name="prediction")

    # === RULES:BEGIN ===
    # ここに条件を記述して prediction にラベルを書き込む
    # 例:
    # cond = (df["列A"] > 10) & (df["列B"].astype(str).str.strip() == "x")
    # prediction.loc[cond] = "製作種別"
    # === RULES:END ===

    return prediction
