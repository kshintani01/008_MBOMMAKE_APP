import io
import os
import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render
from .forms import RulePromptForm, CSVUploadForm
from .services import azure_aoai
from .services.rule_sandbox import run_on_dataframe, SandboxError
from .services.ml_engine import train_quick, predict_with_pipeline, save_pipeline, load_pipeline, MODELS_DIR

def index(request):
    return render(request, "index.html")

def rules(request):
    ctx = {"form": RulePromptForm()}
    if request.method == "POST":
        form = RulePromptForm(request.POST, request.FILES)
        if form.is_valid():
            action = form.cleaned_data.get("action")
            natural = form.cleaned_data.get("natural_language")
            code_text = form.cleaned_data.get("code_text")
            # ★ 目的変数を固定
            target_col = "製作種別"
            csv_file = form.cleaned_data.get("csv_file")

            if action == "generate":
                gen = azure_aoai.generate_code(natural or "")
                ctx.update({
                    "form": form,
                    "generated_code": gen,
                    "message": "コードを生成しました（目的変数は固定：製作種別）"
                })
                return render(request, "rules.html", ctx)

            if action in {"apply_generated", "apply_custom"}:
                code = azure_aoai.generate_code(natural or "") if action == "apply_generated" else (code_text or "")
                if not csv_file:
                    ctx.update({
                        "form": form,
                        "generated_code": code,
                        "error": "CSV を選択してください"
                    })
                    return render(request, "rules.html", ctx)

                try:
                    df = pd.read_csv(csv_file)
                    pred = run_on_dataframe(code, df)

                    # 既存列があれば「predicted_製作種別」を追加、なければ「製作種別」を新規作成
                    if target_col in df.columns:
                        df[f"predicted_{target_col}"] = pred
                    else:
                        df[target_col] = pred

                    # ダウンロード返却
                    buf = io.StringIO()
                    df.to_csv(buf, index=False)
                    resp = HttpResponse(buf.getvalue(), content_type="text/csv")
                    resp["Content-Disposition"] = "attachment; filename=rules_result.csv"
                    return resp

                except SandboxError as e:
                    ctx.update({
                        "form": form,
                        "generated_code": code,
                        "error": f"サンドボックス実行失敗: {e}"
                    })
                except Exception as e:
                    ctx.update({
                        "form": form,
                        "generated_code": code,
                        "error": f"CSV処理に失敗: {e}"
                    })
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