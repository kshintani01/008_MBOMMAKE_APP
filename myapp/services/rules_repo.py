# services/rules_repo.py
import json, hashlib, time, re, os
from dataclasses import dataclass
from typing import Optional, List
from django.conf import settings
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

def _prefix() -> str:
    return settings.RULES_BLOB_PREFIX.rstrip("/") + "/"

def _active_body_blob() -> str:
    return _prefix() + "active/body.py"

def _active_full_blob() -> str:
    return _prefix() + "active/full.py"

def _history_body_blob(ts: str) -> str:
    return _prefix() + f"history/body_{ts}.py"

def _history_full_blob(ts: str) -> str:
    return _prefix() + f"history/full_{ts}.py"

def _ts():
    return time.strftime("%Y%m%d-%H%M%S")

def _blob_client() -> BlobServiceClient:
    # 失敗すれば例外をそのまま上げる（Blob 必須）
    return BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)

def _upload_text(blob_name: str, text: str) -> None:
    bc = _blob_client()
    cli = bc.get_blob_client(settings.AZURE_STORAGE_CONTAINER, blob_name)
    cli.upload_blob(text.encode("utf-8"), overwrite=True)

def _list_history(kind: str, limit: int = 50) -> List[str]:
    """
    kind: "body" or "full"
    returns: blob names sorted desc by timestamp
    """
    bc = _blob_client()
    cont = bc.get_container_client(settings.AZURE_STORAGE_CONTAINER)
    pfx = _prefix() + "history/"
    suffix = f"{kind}_"  # e.g., body_ / full_
    items = []
    for b in cont.list_blobs(name_starts_with=pfx):
        name = b.name
        if name.endswith(".py") and f"/{kind}_" in name:
            items.append(name)
    # 降順ソート（名前の中の時刻で）
    items.sort(reverse=True)
    return items[:limit]

def _blob_name(stem: str, ext: str) -> str:
    # 例: rules/rules_20250916-120000_draft_ab12cd34ef56.py
    prefix = settings.RULES_BLOB_PREFIX.rstrip("/") + "/"
    return f"{prefix}{stem}{ext}"

@dataclass
class RuleMeta:
    tag: str
    sha: str
    timestamp: str
    filename: str   # 保存時のローカル名だが、stem 抽出に使う

def _list_meta_blob_names(bc: BlobServiceClient) -> List[str]:
    container = bc.get_container_client(settings.AZURE_STORAGE_CONTAINER)
    prefix = settings.RULES_BLOB_PREFIX.rstrip("/") + "/"
    # .json だけ列挙
    return sorted(
        [b.name for b in container.list_blobs(name_starts_with=prefix) if b.name.endswith(".json")],
        reverse=True,  # 新しい順（ファイル名の timestamp 前提）
    )

def _download_text(bc: BlobServiceClient, blob_name: str) -> Optional[str]:
    # Try to use account key-based SAS if account credentials are available, otherwise fall back
    account_name = getattr(settings, "AZURE_STORAGE_ACCOUNT_NAME", None) or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = getattr(settings, "AZURE_STORAGE_ACCOUNT_KEY", None) or os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    if account_name and account_key:
        sas = generate_blob_sas(
            account_name=account_name,
            account_key=account_key,
            container_name=settings.AZURE_STORAGE_CONTAINER,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        url = f"https://{account_name}.blob.core.windows.net/{settings.AZURE_STORAGE_CONTAINER}/{blob_name}?{sas}"
        client = BlobClient.from_blob_url(url)
        if not client.exists():
            return None
        return client.download_blob().readall().decode("utf-8")

    client = bc.get_blob_client(settings.AZURE_STORAGE_CONTAINER, blob_name)
    if not client.exists():
        return None
    return client.download_blob().readall().decode("utf-8")

def save_version(code_text: str, *, tag: str = "draft") -> None:
    """
    Blob のみに保存（ローカルへは保存しない）。
    """
    bc = _blob_client()
    sha = hashlib.sha256(code_text.encode("utf-8")).hexdigest()[:12]
    ts = _ts()
    stem = f"rules_{ts}_{tag}_{sha}"

    meta = {
        "tag": tag,
        "sha": sha,
        "timestamp": ts,
        "filename": f"{stem}.py",
    }
    py_name = _blob_name(stem, ".py")
    json_name = _blob_name(stem, ".json")

    bc.get_blob_client(settings.AZURE_STORAGE_CONTAINER, py_name)\
      .upload_blob(code_text.encode("utf-8"), overwrite=True)
    bc.get_blob_client(settings.AZURE_STORAGE_CONTAINER, json_name)\
      .upload_blob(json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"), overwrite=True)

def load_base_code(*, prefer_tag: str = "approved") -> str:
    """
    Blob 専用。指定タグが無ければ FileNotFoundError を投げる。
    フォールバック（ローカル/テンプレ）は一切しない。
    """
    bc = _blob_client()
    for meta_blob in _list_meta_blob_names(bc):
        meta_txt = _download_text(bc, meta_blob)
        if not meta_txt:
            continue
        try:
            meta = json.loads(meta_txt)
        except Exception:
            continue
        if meta.get("tag") != prefer_tag:
            continue

        # stem は filename から拡張子を除いたもの
        filename = meta.get("filename", "")
        if not filename.endswith(".py"):
            continue
        stem = filename[:-3]  # ".py" を除去
        code = _download_text(bc, _blob_name(stem, ".py"))
        if code is not None:
            return code

    # 見つからなければ「エラーにする」要件どおり例外
    raise FileNotFoundError(f"No rule code found on Blob for tag='{prefer_tag}'")

def list_versions(limit: int = 20):
    bc = _blob_client()
    names = _list_meta_blob_names(bc)[:limit]
    # ユーザー表示用に末尾だけ返す
    return [n.split("/")[-1] for n in names]


_BEGIN = re.compile(r"^\s*#\s*===\s*RULES:BEGIN\s*===\s*$", re.M)
_END   = re.compile(r"^\s*#\s*===\s*RULES:END\s*===\s*$", re.M)

def extract_rules_body(full_code: str) -> str:
    """
    full_code から RULES ブロックの“中身だけ”を返す。
    見つからない場合は空文字（または好みで例外）にする。
    """
    m1 = _BEGIN.search(full_code)
    m2 = _END.search(full_code)
    if not (m1 and m2 and m1.end() <= m2.start()):
        return ""  # or raise ValueError("RULES block not found")
    body = full_code[m1.end():m2.start()]
    # 先頭の共通インデントを落とす（4スペ想定）
    lines = [ln.rstrip("\n") for ln in body.splitlines()]
    # 左端トリム（最小の空白幅）
    def lspace(s: str) -> int:
        return len(s) - len(s.lstrip(" "))
    pad = min([lspace(s) for s in lines if s.strip()] or [0])
    norm = "\n".join(s[pad:] for s in lines)
    return norm.strip() + ("\n" if norm.strip() else "")

def inject_rules_body(full_code: str, new_body: str) -> str:
    m1 = _BEGIN.search(full_code)
    m2 = _END.search(full_code)
    if not (m1 and m2 and m1.end() <= m2.start()):
        raise ValueError("RULES block not found in full_code")
    # 既存の body の先頭インデントを推定して維持
    existing = full_code[m1.end():m2.start()]
    # 既存の先頭行の空白幅
    import textwrap
    first_line = next((ln for ln in existing.splitlines() if ln.strip()), "")
    left = len(first_line) - len(first_line.lstrip(" "))
    # 新しい body をそのインデントでインデント
    indented = textwrap.indent(new_body.rstrip("\n") + "\n", " " * left)
    return full_code[:m1.end()] + indented + full_code[m2.start():]

# --- 公開 API -------------------------------------------------------------

def load_active_full_or_error() -> str:
    """
    現在有効な“フル関数”を返す。無ければ FileNotFoundError。
    """
    bc = _blob_client()
    code = _download_text(bc, _active_full_blob())   # ★ 2引数で呼ぶ
    if code is None:
        raise FileNotFoundError("Blob rules/active/full.py が見つかりません。")
    return code

def load_active_body_or_error() -> str:
    """
    現在有効な“ボディ”を返す。無ければ FileNotFoundError。
    """
    bc = _blob_client()
    code = _download_text(bc, _active_full_blob())
    if code is None:
        raise FileNotFoundError("Blob rules/active/body.py が見つかりません。")
    return code

def save_active_and_history(*, full_code: str, body_code: Optional[str] = None) -> str:
    """
    active に full/body を保存し、同時に history にも刻む。
    body_code が None のときは full から抽出して保存。
    returns: 保存に使ったタイムスタンプ文字列
    """
    if body_code is None:
        body_code = extract_rules_body(full_code)

    ts = _ts()
    # active
    _upload_text(_active_full_blob(), full_code)
    _upload_text(_active_body_blob(), body_code)
    # history
    _upload_text(_history_full_blob(ts), full_code)
    _upload_text(_history_body_blob(ts), body_code)
    return ts

def list_history(kind: str = "full", limit: int = 20) -> List[str]:
    """
    kind: "full" or "body"
    returns: blob 名の末尾（ファイル名）のリスト
    """
    names = _list_history(kind, limit)
    return [n.split("/")[-1] for n in names]