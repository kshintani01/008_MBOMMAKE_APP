from django import forms


class RulePromptForm(forms.Form):
    natural_language = forms.CharField(
        label="自然言語のルール", widget=forms.Textarea(attrs={"rows": 8}), required=False
    )
    code_text = forms.CharField(
        label="生成/編集したPythonコード (apply_rules(df) を定義)",
        widget=forms.Textarea(attrs={"rows": 16, "style": "font-family:monospace;"}),
        required=False,
    )
    target_col = forms.CharField(label="目的変数のカラム名", required=False, initial="target")
    csv_file = forms.FileField(label="CSVファイル", required=False)
    action = forms.CharField(widget=forms.HiddenInput(), required=False)


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label="CSVファイル")
    target_col = forms.CharField(label="目的変数のカラム名")
    feature_cols = forms.CharField(
    label="説明変数のカンマ区切り（空なら自動選択）",
    required=False,
        help_text="例: age,income,area_code"
    )
    model_choice = forms.ChoiceField(
    label="モデル",
    choices=[
        ("auto-train", "アップロードCSVで簡易学習して予測"),
        ("load-joblib", "models/ 配下の学習済みモデルを使用"),
    ],
    initial="auto-train",
    )