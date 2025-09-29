from __future__ import annotations
import os
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump, load


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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
    # 目的変数が空白/0/空白文字列を除外（ユーザの既要求に沿う）
    y_raw = df[target_col]
    mask_valid = (~y_raw.isna()) & (y_raw != 0) & (y_raw.astype(str).str.strip() != '')
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

def load_automl_model(model_path: str | os.PathLike):
    """Azure AutoML互換モデルの読み込み"""
    try:
        model_data = load(model_path)
        return model_data
    except Exception as e:
        raise Exception(f"Azure AutoMLモデルの読み込みに失敗: {e}")

def predict_with_automl_model(df: pd.DataFrame, model_data: dict) -> pd.Series:
    """Azure AutoMLモデルで予測"""
    try:
        # モデルコンポーネントを取得
        scaler = model_data['scaler']
        model = model_data['model']
        label_encoder = model_data.get('label_encoder')
        feature_columns = model_data['feature_columns']
        target_column = model_data['target_column']
        
        print(f"AutoMLモデル予測開始: {target_column}")
        
        # データ前処理（学習時と同じ処理）
        df_processed = df.copy()
        
        # 欠損値処理
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(
                    df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else "unknown"
                )
        
        # カテゴリ変数のエンコーディング（学習時のエンコーダーを使用）
        categorical_encoders = model_data.get('categorical_encoders', {})
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if col in categorical_encoders:
                    # 学習時のエンコーダーを使用
                    le = categorical_encoders[col]
                    try:
                        # 未知のカテゴリがある場合の処理
                        df_processed[col] = df_processed[col].astype(str)
                        # 学習時に見たことのないカテゴリを最頻値に置換
                        known_classes = set(le.classes_)
                        mask_unknown = ~df_processed[col].isin(known_classes)
                        if mask_unknown.any():
                            most_frequent = le.classes_[0]  # 最初のクラスを使用
                            df_processed.loc[mask_unknown, col] = most_frequent
                            print(f"警告: 列 '{col}' に未知のカテゴリがあり、'{most_frequent}' に置換しました")
                        
                        df_processed[col] = le.transform(df_processed[col])
                    except Exception as e:
                        print(f"エラー: 列 '{col}' のエンコーディングに失敗: {e}")
                        # フォールバック：新しいエンコーダーを作成
                        le_new = LabelEncoder()
                        df_processed[col] = le_new.fit_transform(df_processed[col].astype(str))
                else:
                    # エンコーダーが見つからない場合は新しく作成
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        # 特徴量の順序を合わせる（学習時と同じ順序）
        try:
            X = df_processed[feature_columns]
        except KeyError as e:
            missing_cols = set(feature_columns) - set(df_processed.columns)
            raise Exception(f"必要な特徴量が不足しています: {missing_cols}")
        
        # 正規化
        X_scaled = scaler.transform(X)
        
        # 予測
        predictions = model.predict(X_scaled)
        
        # ラベルエンコーダーが使用されている場合は逆変換
        if label_encoder and hasattr(label_encoder, 'classes_'):
            try:
                predictions = label_encoder.inverse_transform(predictions)
            except Exception:
                # 逆変換に失敗した場合はそのまま使用
                pass
        
        return pd.Series(predictions, name=f"predicted_{target_column}", index=df.index)
        
    except Exception as e:
        raise Exception(f"Azure AutoMLモデルでの予測に失敗: {e}")