"""
学習済みモデルをAzure Blob Storageにアップロードするスクリプト
"""
import sys
import os
from pathlib import Path
import joblib

# Djangoの設定を読み込み
sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

import django
django.setup()

from myapp.services.blob_storage import get_blob_manager


def upload_automl_model(local_model_path: str, blob_name: str = "automl_model.pkl"):
    """
    ローカルのモデルファイルをBlob Storageにアップロード
    
    Args:
        local_model_path: ローカルのモデルファイルパス
        blob_name: Blob Storage内での名前
    """
    try:
        # ローカルモデルファイルを読み込み
        if not os.path.exists(local_model_path):
            print(f"エラー: モデルファイルが見つかりません: {local_model_path}")
            return False
        
        print(f"ローカルモデルファイルを読み込み中: {local_model_path}")
        model_data = joblib.load(local_model_path)
        
        # Blob Storageにアップロード
        blob_manager = get_blob_manager()
        success = blob_manager.upload_model(model_data, blob_name)
        
        if success:
            print(f"✅ モデルアップロード成功!")
            print(f"   ローカル: {local_model_path}")
            print(f"   Blob: {blob_name}")
            
            # 確認: アップロードされたモデルをダウンロードしてテスト
            print("\n📥 アップロード確認中...")
            downloaded_model = blob_manager.download_model(blob_name)
            if downloaded_model is not None:
                print("✅ アップロード確認成功！")
                print(f"   モデル情報:")
                print(f"     - ターゲット列: {downloaded_model.get('target_column', 'N/A')}")
                print(f"     - 特徴量数: {len(downloaded_model.get('feature_columns', []))}")
                print(f"     - カテゴリエンコーダー数: {len(downloaded_model.get('categorical_encoders', {}))}")
            else:
                print("❌ アップロード確認失敗")
                return False
        else:
            print("❌ モデルアップロード失敗")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return False


def list_blob_models():
    """Blob Storage内のモデル一覧を表示"""
    try:
        blob_manager = get_blob_manager()
        models = blob_manager.list_models()
        
        if models:
            print(f"\n📋 Blob Storage内のモデル一覧 ({len(models)}個):")
            for model in models:
                exists = blob_manager.model_exists(model)
                status = "✅" if exists else "❌"
                print(f"   {status} {model}")
        else:
            print("\n📋 Blob Storage内にモデルが見つかりませんでした")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")


def main():
    """メイン実行関数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python upload_model_to_blob.py <local_model_path> [blob_name]")
        print("  python upload_model_to_blob.py list  # モデル一覧表示")
        print("")
        print("例:")
        print("  python upload_model_to_blob.py models/automl_model.pkl")
        print("  python upload_model_to_blob.py models/automl_model.pkl my_model.pkl")
        print("  python upload_model_to_blob.py list")
        return
    
    if sys.argv[1] == "list":
        list_blob_models()
        return
    
    local_model_path = sys.argv[1]
    blob_name = sys.argv[2] if len(sys.argv) > 2 else "automl_model.pkl"
    
    print(f"🚀 Azure Blob Storageへのモデルアップロード開始")
    print(f"   ローカルファイル: {local_model_path}")
    print(f"   Blob名: {blob_name}")
    print("")
    
    success = upload_automl_model(local_model_path, blob_name)
    
    if success:
        print(f"\n🎉 処理完了！App Serviceから '{blob_name}' でモデルが使用できます。")
    else:
        print(f"\n💥 処理失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()