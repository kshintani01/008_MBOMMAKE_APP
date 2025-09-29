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
    csv_file = forms.FileField(
        label="CSVファイル",
        help_text="AUTO_WTを予測するCSVファイルをアップロードしてください"
    )

class RulePromptForm(forms.Form):
    natural_language = forms.CharField(
        required=False, widget=forms.Textarea(attrs={"rows": 6})
    )
    code_text = forms.CharField(                 # ✅ 必須
        required=False, widget=forms.Textarea(attrs={"rows": 18, "style": "font-family:monospace;width:100%;"})
    )
    csv_file = forms.FileField(required=False)
    action = forms.CharField(widget=forms.HiddenInput(), required=False)