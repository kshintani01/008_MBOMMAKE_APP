# services/rules_repo.py
import json, hashlib, time
from pathlib import Path
from django.conf import settings

def _ts():
    return time.strftime("%Y%m%d-%H%M%S")

def save_version(code_text: str, *, tag: str = "draft") -> Path:
    RULES_DIR = settings.RULES_DIR
    h = hashlib.sha256(code_text.encode("utf-8")).hexdigest()[:12]
    stem = f"rules_{_ts()}_{tag}_{h}"
    py_path = RULES_DIR / f"{stem}.py"
    meta_path = RULES_DIR / f"{stem}.json"

    py_path.write_text(code_text, encoding="utf-8")
    meta = {
        "tag": tag,
        "sha": h,
        "timestamp": _ts(),
        "filename": py_path.name,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return py_path

def load_base_code(prefer_tag: str = "approved") -> str:
    """承認済み（approved）を優先。なければテンプレートを返す。"""
    RULES_DIR = settings.RULES_DIR
    metas = sorted(RULES_DIR.glob("rules_*.json"), reverse=True)
    for m in metas:
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("tag") == prefer_tag:
            py = RULES_DIR / meta.get("filename", "")
            if py.exists():
                return py.read_text(encoding="utf-8")

    # fallback: テンプレート
    tpl = getattr(settings, "RULES_TEMPLATE_PATH", None)
    if tpl and Path(tpl).exists():
        return Path(tpl).read_text(encoding="utf-8")

    # 最後の保険（絶対に空にしない）
    return (
        "import pandas as pd\nimport numpy as np\n\n"
        "def apply_rules(df):\n"
        "    prediction = pd.Series([None]*len(df), name='prediction')\n\n"
        "    # === RULES:BEGIN ===\n"
        "    # TODO: ここにルールを記述\n"
        "    # === RULES:END ===\n\n"
        "    return prediction\n"
    )

def list_versions(limit: int = 20):
    RULES_DIR = settings.RULES_DIR
    items = sorted(RULES_DIR.glob("rules_*.py"), reverse=True)
    return [p.name for p in items[:limit]]
