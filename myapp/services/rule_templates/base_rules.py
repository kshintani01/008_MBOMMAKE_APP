import pandas as pd
import numpy as np

def apply_rules(df):
    prediction = pd.Series(index=df.index, dtype=object, name="prediction")

    # === RULES:BEGIN ===
    # （ここに生成AIが書いた条件代入が入る）
    # === RULES:END ===

    # もし BEGIN〜END の中で prediction.loc[...] が一度も出てこなければ
    # デフォルトで「未分類」を代入する
    if (prediction.isna()).all():
        prediction[:] = "未分類"

    return prediction