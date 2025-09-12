# rules_store.py
import os, io, datetime as dt
from typing import Optional
from azure.storage.blob import BlobServiceClient, BlobClient
import pandas as pd  # 既存コードに合わせてインポート可能

BLOB_CONN_STR  = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

def _svc():
    if not (BLOB_CONN_STR and BLOB_CONTAINER):
        raise RuntimeError("Blob接続情報が未設定です")
    return BlobServiceClient.from_connection_string(BLOB_CONN_STR)

def _blob(path: str) -> BlobClient:
    return _svc().get_blob_client(container=BLOB_CONTAINER, blob=path)

def save_text(path: str, text: str) -> None:
    bc = _blob(path)
    bc.upload_blob(text.encode("utf-8"), overwrite=True)

def load_text(path: str) -> Optional[str]:
    bc = _blob(path)
    if not bc.exists():
        return None
    return bc.download_blob().readall().decode("utf-8")

def now_tag():
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

# 保存API
def save_rules_body(body: str):
    ts = now_tag()
    save_text(f"rules/history/body_{ts}.py", body)
    save_text("rules/active/body.py", body)

def save_rules_full(code: str):
    ts = now_tag()
    save_text(f"rules/history/full_{ts}.py", code)
    save_text("rules/active/full.py", code)

# 読み込みAPI
def load_rules_body() -> Optional[str]:
    return load_text("rules/active/body.py")

def load_rules_full() -> Optional[str]:
    return load_text("rules/active/full.py")
