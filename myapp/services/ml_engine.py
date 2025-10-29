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

from .text_normalizer import normalize_df_kana

# Azure Blob Storage サポート
try:
    from .blob_storage import get_blob_manager
    from .aml_endpoint import AMLConfig, score_via_endpoint
    BLOB_STORAGE_AVAILABLE = True
except ImportError:
    BLOB_STORAGE_AVAILABLE = False
    print("警告: Azure Blob Storageが利用できません。ローカルファイルシステムを使用します。")

# Django設定からMODELS_DIRを取得（Docker環境対応）
try:
    from django.conf import settings
    MODELS_DIR = getattr(settings, 'MODELS_DIR', Path("models"))
except ImportError:
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)

class TrainResult:
    def __init__(self, pipeline, classes, f1=None):
        self.pipeline = pipeline
        self.classes = classes
        self.f1 = f1

def predict_auto(df_in: pd.DataFrame) -> pd.DataFrame:
    """環境変数 USE_AZURE_ML_ENDPOINT=1 ならエンドポイント推論、それ以外はローカル推論。常にNFKC正規化。"""
    df = normalize_df_kana(df_in.copy())
    use_aml = os.getenv("USE_AZURE_ML_ENDPOINT", "0") in ("1", "true", "True")
    if use_aml and os.getenv("AML_SCORING_URI"):
        return predict_with_azure_endpoint(df)
    return predict_with_pipeline(df)

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


def predict_with_pipeline(
    df: pd.DataFrame,
    target_col=None,           # ← 省略可にする
    pipeline=None              # ← 省略可にする
) -> pd.DataFrame:
    """
    互換ラッパ：
    - pipeline/target_col を渡されたらそれを使う
    - 渡されなければ、従来の AutoML モデル（ローカル or Blob）を自動ロードして推論
    """
    # pipeline が渡された場合はそれを使う
    if pipeline is not None:
        preds = pipeline.predict(df)
        pred_col = target_col or os.getenv("ML_PREDICTION_COL", "prediction")
        out = df.copy()
        out[pred_col] = preds
        return out

    # ここから “従来の AutoML モデルを自動ロード” ルート
    models_dir = getattr(settings, "MODELS_DIR", Path(settings.BASE_DIR) / "models")
    local_fname = os.getenv("AUTOML_LOCAL_FILENAME", "automl_model.pkl")
    blob_name   = os.getenv("AUTOML_BLOB_NAME", "automl_model.pkl")
    local_path  = Path(models_dir) / local_fname

    # 既存のローダと互換に
    model_data = load_automl_model(
        model_path=local_path if local_path.exists() else None,
        blob_name=blob_name
    )
    preds = predict_with_automl_model(df, model_data)
    tgt = model_data.get("target_column", os.getenv("ML_TARGET_COL", "prediction"))

    out = df.copy()
    out[f"predicted_{tgt}"] = preds
    return out


def save_pipeline(result: TrainResult, path: str | os.PathLike):
    dump({"pipeline": result.pipeline, "classes": result.classes}, path)

def load_pipeline(path: str | os.PathLike):
    obj = load(path)
    return obj["pipeline"], obj.get("classes")

def load_automl_model(model_path: str | os.PathLike = None, blob_name: str = "automl_model.pkl"):
    """
    Azure AutoML互換モデルの読み込み
    Blob Storageを優先し、フォールバックでローカルファイルを使用
    
    Args:
        model_path: ローカルファイルパス（フォールバック用）
        blob_name: Blob Storage内のモデル名
    """
    try:
        # Blob Storageから読み込みを試行
        if BLOB_STORAGE_AVAILABLE:
            blob_manager = get_blob_manager()
            model_data = blob_manager.download_model(blob_name)
            if model_data is not None:
                print(f"Blob Storageからモデルを読み込みました: {blob_name}")
                return model_data
            else:
                print(f"Blob Storageにモデルが見つかりません: {blob_name}")
        
        # フォールバック: ローカルファイルから読み込み
        if model_path is not None and os.path.exists(model_path):
            model_data = load(model_path)
            print(f"ローカルファイルからモデルを読み込みました: {model_path}")
            return model_data
        
        raise Exception("モデルファイルが見つかりません（Blob Storage・ローカル共に）")
        
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

def predict_with_azure_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    """新規: Azure ML オンライン推論エンドポイント経由の推論"""
    cfg = AMLConfig.from_env()
    result, meta = score_via_endpoint(df, cfg)
    # 必要なら meta をログに
    print("AML expected cols (from template):", meta.get("expected_cols", [])[:30])
    return result
