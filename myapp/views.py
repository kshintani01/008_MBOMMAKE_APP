import io
import os
import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from .forms import RulePromptForm, CSVUploadForm
from .services import azure_aoai
from .services.rule_sandbox import run_on_dataframe, SandboxError
from .services.ml_engine import train_quick, predict_with_pipeline, save_pipeline, load_pipeline, MODELS_DIR
from .services.rule_patcher import replace_rules_block, unified_diff
from .services.rules_repo import load_active_full_or_error, save_active_and_history, extract_rules_body

def index(request):
    return render(request, "index.html")

def rules(request):
    if request.method == "GET":
        try:
            code = load_active_full_or_error()  # ★ Blob の active/full.py を読む
            form = RulePromptForm(initial={"code_text": code})
            return render(request, "rules.html", {"form": form})
        except Exception as e:
            form = RulePromptForm(initial={"code_text": ""})
            return render(request, "rules.html", {"form": form, "error": f"アクティブなルールを取得できません: {e}"})

    # POST
    form = RulePromptForm(request.POST, request.FILES)
    action = request.POST.get("action")
    ctx = {"form": form}

    if not form.is_valid():
        ctx["error"] = "入力を確認してください"
        return render(request, "rules.html", ctx)

    natural = form.cleaned_data.get("natural_language") or ""
    code_text = form.cleaned_data.get("code_text") or ""   # ここには“フル関数”が入ってくる想定
    csv_file = form.cleaned_data.get("csv_file")
    target_col = "製作種別"

    if action == "generate_block":
        try:
            rules_body_llm = azure_aoai.generate_rules_body(natural, code_text)
            rules_body = azure_aoai.choose_rules_body(rules_body_llm)
            new_code = replace_rules_block(code_text, rules_body)
            diff_text = unified_diff(code_text, new_code, "base.py", "new.py")
            ctx.update({"form": form, "generated_code": new_code, "diff_text": diff_text, "message": "差分を生成しました。"})
        except Exception as e:
            ctx.update({"form": form, "generated_code": code_text, "error": f"差分生成に失敗しました: {e}"})
        return render(request, "rules.html", ctx)

    if action == "apply_to_editor":
        try:
            rules_body_llm = azure_aoai.generate_rules_body(natural, code_text)
            rules_body = azure_aoai.choose_rules_body(rules_body_llm)
            new_code = replace_rules_block(code_text, rules_body)

            body_for_save = extract_rules_body(new_code) or rules_body
            save_active_and_history(full_code=new_code, body_code=body_for_save)

            form = RulePromptForm(initial={"natural_language": natural, "code_text": new_code})
            ctx = {"form": form, "generated_code": new_code, "message": "差分をエディタに反映しました。"}
        except Exception as e:
            ctx.update({"form": form, "error": f"反映に失敗: {e}"})
        return render(request, "rules.html", ctx)

    if action == "apply_and_run":
        if not csv_file:
            ctx.update({"form": form, "error": "CSV を選択してください"})
            return render(request, "rules.html", ctx)
        try:
            # 1) ボディ生成＋フルへ埋め込み
            rules_body_llm = azure_aoai.generate_rules_body(natural, code_text)
            rules_body = azure_aoai.choose_rules_body(rules_body_llm)
            new_code = replace_rules_block(code_text, rules_body)

            # 2) 実行（サンドボックス）
            df = pd.read_csv(csv_file)
            pred = run_on_dataframe(new_code, df)

            if target_col in df.columns:
                df[f"predicted_{target_col}"] = pred
            else:
                df[target_col] = pred

            # 3) ★ 保存：active と history の両方に保存（Blob のみ）
            save_active_and_history(full_code=new_code, body_code=rules_body)

            # 4) ダウンロード返却
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            resp = HttpResponse(buf.getvalue(), content_type="text/csv")
            resp["Content-Disposition"] = "attachment; filename=rules_result.csv"
            return resp

        except SandboxError as e:
            ctx.update({"form": form, "error": f"サンドボックス実行失敗: {e}"})
        except Exception as e:
            ctx.update({"form": form, "error": f"処理に失敗: {e}"})
        return render(request, "rules.html", ctx)

    return render(request, "rules.html", ctx)

def ml(request):
    ctx = {"form": CSVUploadForm()}
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csvf = form.cleaned_data["csv_file"]
            target_col = form.cleaned_data["target_col"]
            feature_cols_raw = form.cleaned_data.get("feature_cols")
            feature_cols = [c.strip() for c in feature_cols_raw.split(",") if c.strip()] if feature_cols_raw else None
            model_choice = form.cleaned_data["model_choice"]

            try:
                df = pd.read_csv(csvf)
                if target_col not in df.columns:
                    return render(request, "ml.html", {"form": form, "error": f"目的変数 {target_col} がCSVにありません"})

                if model_choice == "load-joblib":
                    path = MODELS_DIR / "xgb_std_pipeline.joblib"
                    if not path.exists():
                        return render(request, "ml.html", {"form": form, "error": f"{path} が見つかりません。auto-train を選択してください。"})
                    pipe, _ = load_pipeline(path)
                else:
                    # アップロードCSVで簡易学習
                    tr = train_quick(df.copy(), target_col, feature_cols)
                    pipe = tr.pipeline
                    save_pipeline(tr, MODELS_DIR / "xgb_std_pipeline.joblib")


                pred = predict_with_pipeline(df, target_col, pipe)
                out = df.copy()
                out[f"predicted_{target_col}"] = pred


                buf = io.StringIO()
                out.to_csv(buf, index=False)
                resp = HttpResponse(buf.getvalue(), content_type="text/csv")
                resp["Content-Disposition"] = "attachment; filename=ml_result.csv"
                return resp


            except Exception as e:
                return render(request, "ml.html", {"form": form, "error": f"エラー: {e}"})

    return render(request, "ml.html", ctx)

def aoai_check(request):
    try:
        txt = azure_aoai.healthcheck()
        ok = "ok" in txt.lower()
        return JsonResponse({"success": ok, "echo": txt}, status=200 if ok else 502)
    except Exception as e:
        return JsonResponse({"success": False, "error":str(e)}, status=502)