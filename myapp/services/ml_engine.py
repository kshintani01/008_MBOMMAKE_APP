from __future__ import annotations
import os
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump, load


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

class TrainResult:
    def __init__(self, pipeline, classes, f1=None):
        self.pipeline = pipeline
        self.classes = classes
        self.f1 = f1

def _infer_columns(df: pd.DataFrame, target_col: str, feature_cols: list[str] | None):
    if feature_cols:
        feats = feature_cols
    else:
        feats = [c for c in df.columns if c != target_col]
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feats if c not in num_cols]
    return num_cols, cat_cols

def train_quick(df: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None) -> TrainResult:
    # 目的変数が空白/0 を除外（ユーザの既要求に沿う）
    y_raw = df[target_col]
    mask_valid = (~y_raw.isna()) & (y_raw != 0)
    train = df.loc[mask_valid].copy()

    y = train[target_col]
    X = train.drop(columns=[target_col])

    num_cols, cat_cols = _infer_columns(df, target_col, feature_cols)

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    ct = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    model = LogisticRegression(max_iter=200)

    pipe = Pipeline([
        ("prep", ct),
        ("clf", model),
    ])

    pipe.fit(X[num_cols + cat_cols], y)

    # 簡易 F1（学習データ）
    try:
        y_pred = pipe.predict(X[num_cols + cat_cols])
        micro_f1 = f1_score(y, y_pred, average="micro")
    except Exception:
        micro_f1 = None

    return TrainResult(pipe, classes=getattr(model, "classes_", None), f1=micro_f1)




def predict_with_pipeline(df: pd.DataFrame, target_col: str, pipeline: Pipeline) -> pd.Series:
    feats = [c for c in df.columns if c != target_col]
    pred = pipeline.predict(df[feats])
    s = pd.Series(pred, name=f"predicted_{target_col}", index=df.index)
    return s

def save_pipeline(result: TrainResult, path: str | os.PathLike):
    dump({"pipeline": result.pipeline, "classes": result.classes}, path)

def load_pipeline(path: str | os.PathLike):
    obj = load(path)
    return obj["pipeline"], obj.get("classes")