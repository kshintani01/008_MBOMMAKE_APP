import pandas as pd 
import numpy as np

def apply_rules(df):
    prediction = pd.Series([None] * len(df), name="prediction")

    # === RULES:BEGIN ===
    # ここに条件を記述いs亭 prediction にラベルを書き込む
    # 例:
    # cond = (df["列A"] > 10) & (df["列B"] == "x")
    # prediction.loc[cond] = "製作種別"
    # === RULES:END ===

    return prediction

