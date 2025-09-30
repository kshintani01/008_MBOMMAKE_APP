"""
Azure Blob Storage用のヘルパー関数
MLモデルの保存・読み込みを行う
"""
import os
import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import joblib
from django.conf import settings


class BlobStorageModelManager:
    """Azure Blob StorageでMLモデルを管理するクラス"""
    
    def __init__(self):
        # 設定から接続文字列を取得（Managed Identityがない場合のフォールバック）
        self.connection_string = getattr(settings, 'AZURE_STORAGE_CONNECTION_STRING', None)
        self.account_name = getattr(settings, 'AZURE_STORAGE_ACCOUNT_NAME', None)
        self.container_name = getattr(settings, 'ML_MODELS_CONTAINER', 'your-container')
        self.blob_prefix = getattr(settings, 'ML_MODELS_BLOB_PREFIX', 'models/')
        
        # BlobServiceClientを初期化
        if self.connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        elif self.account_name:
            # Managed Identity使用（推奨）
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        else:
            raise ValueError("Azure Storage設定が不足しています。AZURE_STORAGE_CONNECTION_STRING または AZURE_STORAGE_ACCOUNT_NAME を設定してください。")
        
        print(f"MLモデル設定: コンテナ={self.container_name}, プレフィックス={self.blob_prefix}")
    
    def upload_model(self, model_data: Dict[str, Any], blob_name: str) -> bool:
        """
        モデルデータをBlob Storageにアップロード
        
        Args:
            model_data: 保存するモデルデータ
            blob_name: Blob名（例: 'automl_model.pkl'）
            
        Returns:
            bool: 成功/失敗
        """
        try:
            # メモリ上でモデルをシリアライズ
            buffer = io.BytesIO()
            joblib.dump(model_data, buffer)
            buffer.seek(0)
            
            # Blobにアップロード（プレフィックス付き）
            full_blob_name = f"{self.blob_prefix}{blob_name}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=full_blob_name
            )
            
            blob_client.upload_blob(buffer.getvalue(), overwrite=True)
            print(f"モデルをBlob Storageにアップロードしました: {full_blob_name}")
            return True
            
        except Exception as e:
            print(f"モデルアップロードエラー: {e}")
            return False
    
    def download_model(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """
        Blob Storageからモデルデータをダウンロード
        
        Args:
            blob_name: Blob名（例: 'automl_model.pkl'）
            
        Returns:
            Dict[str, Any]: モデルデータ（失敗時はNone）
        """
        try:
            full_blob_name = f"{self.blob_prefix}{blob_name}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=full_blob_name
            )
            
            # Blobデータをメモリに読み込み
            blob_data = blob_client.download_blob().readall()
            
            # バイナリデータからモデルを復元
            buffer = io.BytesIO(blob_data)
            model_data = joblib.load(buffer)
            
            print(f"モデルをBlob Storageからダウンロードしました: {full_blob_name}")
            return model_data
            
        except ResourceNotFoundError:
            print(f"モデルファイルが見つかりません: {blob_name}")
            return None
        except Exception as e:
            print(f"モデルダウンロードエラー: {e}")
            return None
    
    def download_model_to_cache(self, blob_name: str, cache_path: Optional[Path] = None) -> Optional[Path]:
        """
        Blob Storageからモデルをローカルキャッシュにダウンロード
        
        Args:
            blob_name: Blob名
            cache_path: キャッシュパス（Noneの場合は一時ディレクトリ）
            
        Returns:
            Path: ダウンロードしたファイルのパス（失敗時はNone）
        """
        try:
            if cache_path is None:
                cache_dir = Path(tempfile.gettempdir()) / "azure_models"
                cache_dir.mkdir(exist_ok=True)
                cache_path = cache_dir / blob_name
            
            full_blob_name = f"{self.blob_prefix}{blob_name}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=full_blob_name
            )
            
            # ファイルに直接ダウンロード
            with open(cache_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            print(f"モデルをキャッシュにダウンロードしました: {cache_path}")
            return cache_path
            
        except Exception as e:
            print(f"キャッシュダウンロードエラー: {e}")
            return None
    
    def list_models(self) -> list:
        """
        Blob Storage内のモデル一覧を取得
        
        Returns:
            list: モデルファイル名のリスト
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = container_client.list_blobs(name_starts_with=self.blob_prefix)
            # プレフィックスを除いたファイル名を返す
            return [blob.name[len(self.blob_prefix):] for blob in blob_list if blob.name.endswith('.pkl')]
        except Exception as e:
            print(f"モデル一覧取得エラー: {e}")
            return []
    
    def model_exists(self, blob_name: str) -> bool:
        """
        モデルがBlob Storageに存在するかチェック
        
        Args:
            blob_name: Blob名
            
        Returns:
            bool: 存在する/しない
        """
        try:
            full_blob_name = f"{self.blob_prefix}{blob_name}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=full_blob_name
            )
            return blob_client.exists()
        except Exception:
            return False


# グローバルインスタンス（シングルトンパターン）
_blob_manager = None

def get_blob_manager() -> BlobStorageModelManager:
    """BlobStorageModelManagerのシングルトンインスタンスを取得"""
    global _blob_manager
    if _blob_manager is None:
        _blob_manager = BlobStorageModelManager()
    return _blob_manager