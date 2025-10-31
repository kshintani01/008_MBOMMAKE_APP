import io, os, unicodedata, re
import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.shortcuts import render
from .forms import RulePromptForm, CSVUploadForm
from .services import azure_aoai, rules_repo
from .services.rule_sandbox import run_on_dataframe, SandboxError
from .services.ml_engine import predict_auto
from .services.rule_patcher import replace_rules_block, unified_diff, merge_rules_body_dedup
from .services.rules_repo import load_active_full_or_error, save_active_and_history, extract_rules_body
from .services.validators import warn_unknown_columns
from django.conf import settings

XREF1_LABEL = getattr(settings, "XREF1_DISPLAY_NAME", "品質定義.xlsx")
XREF2_LABEL = getattr(settings, "XREF2_DISPLAY_NAME", "部材アセットマスタ.xlsx")

def _with_schema(ctx: dict) -> dict:
    """テンプレに固定列リストと active_code を常に渡す"""
    ctx = dict(ctx or {})
    ctx.setdefault("REFERENCE_CSV_01_COLUMNS", getattr(settings, "REFERENCE_CSV_01_COLUMNS", []))
    # 保存済みコード（無ければ空文字）
    try:
      ctx.setdefault("active_code", rules_repo.get_active_full_code())
    except Exception:
      ctx.setdefault("active_code", "")
    return ctx

def index(request):
    return render(request, "index.html")

def rules(request):
    if request.method == "GET":
        try:
            code = load_active_full_or_error()  # ★ Blob の active/full.py を読む
            form = RulePromptForm(initial={"code_text": code})
            return render(request, "rules.html", _with_schema({"form": form, "editor_code": code, "active_code": code}))       
        except Exception as e:
            form = RulePromptForm(initial={"code_text": ""})
            return render(request, "rules.html", _with_schema({"form": form, "error": f"アクティブなルールを取得できません: {e}"}))
        

    # POST
    form = RulePromptForm(request.POST, request.FILES)
    action = request.POST.get("action")
    ctx = {"form": form,
           "xref1_label": XREF1_LABEL,
           "xref2_label": XREF2_LABEL}

    if not form.is_valid():
        ctx["error"] = "入力を確認してください"
        return render(request, "rules.html", _with_schema(ctx))

    natural = form.cleaned_data.get("natural_language") or ""
    code_text = form.cleaned_data.get("code_text") or ""   # ここには“フル関数”が入ってくる想定
    csv_file = form.cleaned_data.get("csv_file")
    target_col = "製作種別"

    if action == "generate_block":
        try:
            add_body = azure_aoai.generate_rules_body(natural, code_text)  # 追加分だけ
            # 空ならそのまま返す
            if not (add_body or "").strip():
                ctx.update({"form": form, "generated_code": code_text, "message": "追加はありませんでした。"})
                return render(request, "rules.html", _with_schema(ctx))

            existing_body = extract_rules_body(code_text) or ""
            merged_body   = merge_rules_body_dedup(existing_body, add_body)  # ★ 重複除外でマージ
            new_code      = replace_rules_block(code_text, merged_body)      # ★ 置換はマージ後の本文で

            diff_text = unified_diff(
                existing_body,               # 旧：ルール本文のみ
                merged_body,                 # 新：ルール本文のみ
                "RULES_BODY(old)",
                "RULES_BODY(new)",
                context=0                    # ← 変更行だけ（コンテキストなし）
            )
            # ★ 追加分を hidden に保持（反映時に使う）
            form = RulePromptForm(initial={"natural_language": natural, "code_text": new_code})
            ctx.update({
                "form": form,
                "generated_code": new_code,
                "diff_text": diff_text,
                "generated_addition": add_body,        # ★ hidden 用
                "message": "差分をエディタに反映しました（未保存）。"
            })
        except Exception as e:
            ctx.update({"form": form, "generated_code": code_text, "error": f"差分生成に失敗しました: {e}"})
        return render(request, "rules.html", _with_schema(ctx))

    if action == "apply_to_editor":
        try:
            # ★ 直前プレビューに関係なく、常に最新版から始める
            latest_full = load_active_full_or_error()
            existing_body = extract_rules_body(latest_full) or ""

            add_body = request.POST.get("generated_addition", "")  # ★ 生成時に hidden に入れておく
            if not (add_body or "").strip():
                ctx.update({"form": form, "error": "追加分が見つかりません。まずは『差分を生成』を行ってください。"})
                return render(request, "rules.html", _with_schema(ctx))

            merged_body = merge_rules_body_dedup(existing_body, add_body)  # ★ 重複除外でマージ
            new_full    = replace_rules_block(latest_full, merged_body)

            # 保存
            save_active_and_history(full_code=new_full, body_code=merged_body)

            form = RulePromptForm(initial={"natural_language": natural, "code_text": new_full})
            ctx = {"form": form, "generated_code": new_full, "message": "最新ルールに追加分をマージして保存しました。"}
        except Exception as e:
            ctx.update({"form": form, "error": f"反映に失敗: {e}"})
        return render(request, "rules.html", _with_schema(ctx))


    if action == "apply_and_run":
        if not csv_file:
            ctx.update({"form": form, "error": "CSV を選択してください"})
            return render(request, "rules.html", _with_schema(ctx))
        try:
            # ① Blobの active/full.py を取得（＝最新の保存版を使う）
            latest_full_code = load_active_full_or_error()

            # ② CSV読込（BOM対策でutf-8-sig、必要ならエラー処理）
            df = _robust_read_csv(csv_file)

            # ★ 列名の表記ゆれを正規化
            df = _normalize_columns_for_rules(df)

            # ★ ルールが参照する古い/揺れた表記名にも対応（影武者列を作成）
            df = _ensure_rule_column_aliases(df, latest_full_code)

            # 正規化後に unknown 列警告を評価
            msgs = warn_unknown_columns(df)
            for m in msgs:
                messages.warning(request, m)

            # ③ サンドボックスで apply_rules(df) 実行
            pred = run_on_dataframe(latest_full_code, df)

            # ④ 出力列を付与して返却（既存の target_col がある/ないを吸収）
            target_col = "製作種別"
            out = df.copy()
            out[f"predicted_{target_col}"] = pred

            buf = io.StringIO()
            return _csv_response_utf8_bom(out, "rules_result.csv")

        except SandboxError as e:
            ctx.update({"form": form, "error": f"サンドボックスエラー: {e}"})
            return render(request, "rules.html", ctx)
        except Exception as e:
            ctx.update({"form": form, "error": f"エラー: {e}"})
            return render(request, "rules.html", ctx)

    return render(request, "rules.html", ctx)

def ml(request):
    ctx = {"form": CSVUploadForm()}
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csvf = form.cleaned_data["csv_file"]
            try:
                df = _robust_read_csv(csvf)
                out = predict_auto(df)
                return _csv_response_utf8_bom(out, "ml_prediction_result.csv")
            except Exception as e:
                return render(request, "ml.html", {"form": form, "error": f"予測に失敗しました: {e}"})
    return render(request, "ml.html", ctx)

def aoai_check(request):
    try:
        txt = azure_aoai.healthcheck()
        ok = "ok" in txt.lower()
        return JsonResponse({"success": ok, "echo": txt}, status=200 if ok else 502)
    except Exception as e:
        return JsonResponse({"success": False, "error":str(e)}, status=502)

def _normalize_columns_for_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    列名をNFKC正規化（全角→半角、互換分解）＋トリムし、
    代表名（canonical）に寄せる。代表名は実データ上の列名として残す。
    例: ユニットＩＤ/ﾕﾆｯﾄID → ユニットID, 部品ｺｰﾄﾞ → 部品コード
    """
    CANON = {
        "ユニットID": {
            "ﾕﾆｯﾄID", "ﾕﾆｯﾄＩＤ", "ユニットＩＤ",
            "ユニット Id", "ﾕﾆｯﾄ Id", "ユニットid", "ﾕﾆｯﾄid",
            "ユニット ID", "ﾕﾆｯﾄ ID", "ユニット　ID", "ﾕﾆｯﾄ　ID",
        },
        "部品コード": {
            "部品ｺｰﾄﾞ", "部品 ｺｰﾄﾞ", "部品 ｺ-ﾄﾞ",
            "部品 コード", "部品  コード",
            "部品CD", "品番", "部品番号",  # 必要に応じて調整
        },
    }

    def canon(s: str) -> str:
        if s is None:
            return ""
        # NFKCで幅・互換文字を揃え、前後空白除去
        return unicodedata.normalize("NFKC", str(s)).strip()

    df2 = df.copy()
    # まず全列をNFKC化
    df2.columns = [canon(c) for c in df2.columns]

    # 同義語→代表名のリネームマップ
    rename_map = {}
    for rep, aliases in CANON.items():
        rep_c = canon(rep)
        rename_map[rep_c] = rep_c  # 自分自身（冪等性）
        for a in aliases:
            rename_map[canon(a)] = rep_c

    # 実際に存在するものだけリネーム
    df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns}, inplace=True)
    return df2

def _ensure_rule_column_aliases(df: pd.DataFrame, rules_code: str) -> pd.DataFrame:
    """
    ルールコード中で参照されている列名（例: df['部品ｺｰﾄﾞ']）を抽出し、
    そのNFKC正規化版（例: 部品コード）がDFに存在するなら、
    参照名の“影武者列”を作ってエイリアス化する（df['部品ｺｰﾄﾞ'] = df['部品コード']）。
    これでコード側が古い/揺れた表記でも壊れない。
    """
    def canon(s: str) -> str:
        return unicodedata.normalize("NFKC", str(s)).strip()

    # df['列名'] / df["列名"] / .loc[:, '列名'] / .loc[:, "列名"] を素直に拾う
    pat = r"(?:df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]|\.loc\s*\[\s*[:,\s]*['\"]([^'\"]+)['\"]\s*\])"
    used = set()
    for m in re.finditer(pat, rules_code or ""):
        col = m.group(1) or m.group(2)
        if col:
            used.add(col)

    if not used:
        return df

    df2 = df.copy()
    canon_cols = {canon(c): c for c in df2.columns}  # NFKC→実列名

    for raw in used:
        raw_c = canon(raw)
        # 既にその名前がDFにあれば何もしない
        if raw in df2.columns:
            continue
        # 正規化名がDFにあれば “影武者列” を作る
        if raw_c in canon_cols:
            src = canon_cols[raw_c]
            try:
                df2[raw] = df2[src]
            except Exception:
                # 安全側でスキップ
                pass

    return df2

def _robust_read_csv(uploaded_file) -> pd.DataFrame:
    """UTF-8(BOM付/無し)→CP932→UTF-8の順で解釈して読み込む"""
    import io as _io
    data = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(_io.BytesIO(data), encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    # 最終フォールバック
    return pd.read_csv(_io.BytesIO(data), low_memory=False)

def _csv_response_utf8_bom(df: pd.DataFrame, filename: str) -> HttpResponse:
    import io as _io
    b = _io.BytesIO()
    # ← ここを lineterminator に変更
    df.to_csv(b, index=False, encoding="utf-8-sig", lineterminator="\r\n")
    resp = HttpResponse(b.getvalue(), content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp