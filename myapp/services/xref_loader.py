# myapp/services/xref_loader.py
import os
from io import BytesIO
import pandas as pd
from functools import lru_cache

from azure.storage.blob import BlobServiceClient

BLOB_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

# 環境変数で Blob 上のパス（blob名）を指定
XREF1_BLOB = os.getenv("AZURE_STORAGE_REFERENCE_CSV_01")  # 例: "refs/parts_master.xlsx"
XREF2_BLOB = os.getenv("AZURE_STORAGE_REFERENCE_CSV_02")  # 例: "refs/spec_mapping.xlsx"

def _download_blob_to_bytes(container: str, blob_name: str) -> bytes:
    if not (BLOB_CONN_STR and container and blob_name):
        raise RuntimeError("Blob settings are not complete.")
    svc = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    data = svc.get_container_client(container).download_blob(blob_name).readall()
    return data

@lru_cache(maxsize=1)
def load_xref1() -> pd.DataFrame:
    """部品マスタなど：XREF1_BLOB"""
    buf = _download_blob_to_bytes(BLOB_CONTAINER, XREF1_BLOB)
    return pd.read_excel(BytesIO(buf))

@lru_cache(maxsize=1)
def load_xref2() -> pd.DataFrame:
    """仕様→区分のマッピングなど：XREF2_BLOB"""
    buf = _download_blob_to_bytes(BLOB_CONTAINER, XREF2_BLOB)
    return pd.read_excel(BytesIO(buf))
