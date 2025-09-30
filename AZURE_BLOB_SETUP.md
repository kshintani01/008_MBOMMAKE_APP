# Azure Blob Storage設定ガイド

## 概要
Azure App ServiceからBlob Storageに保存されたMLモデルを読み込むための設定手順です。

## 必要な環境変数

### 方法1: 接続文字列を使用（開発環境推奨）
```bash
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net"
AZURE_STORAGE_CONTAINER="your-container"
ML_MODELS_BLOB_PREFIX="models/"
```

### 方法2: Managed Identityを使用（本番環境推奨）
```bash
AZURE_STORAGE_ACCOUNT_NAME="your_storage_account"
# 既存のコンテナを使用（ルール保存等と共通）
AZURE_STORAGE_CONTAINER="your-container"
# MLモデル専用フォルダ（オプション、デフォルト: models/）
ML_MODELS_BLOB_PREFIX="models/"
```

## 設定手順

### 1. Azure Storage Accountの作成
```bash
# Resource Group作成
az group create --name myResourceGroup --location "Japan East"

# Storage Account作成
az storage account create \
  --name mystorageaccount \
  --resource-group myResourceGroup \
  --location "Japan East" \
  --sku Standard_LRS
```

### 2. コンテナの設定
```bash
# 既存のコンテナを使用（例: your-container）
# その中にmodels/フォルダが自動作成されます
# 新しいコンテナが必要な場合：
az storage container create \
  --name your-container \
  --account-name mystorageaccount
```

### 3. Managed Identityの設定（本番環境）
```bash
# App ServiceでManaged Identityを有効化
az webapp identity assign --name myappservice --resource-group myResourceGroup

# Storage BlobデータContributorロールを付与
az role assignment create \
  --assignee <managed-identity-principal-id> \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/<subscription-id>/resourceGroups/myResourceGroup/providers/Microsoft.Storage/storageAccounts/mystorageaccount"
```

## モデルのアップロード

### 1. ローカルでモデルをアップロード
```bash
# 学習済みモデルをBlob Storageにアップロード
# コンテナ/models/automl_model.pkl として保存されます
python upload_model_to_blob.py models/automl_model.pkl

# モデル一覧確認（models/フォルダ内）
python upload_model_to_blob.py list
```

### 2. Blob Storage内の構造
```
your-container/
├── rules/                    # ルール保存用（既存）
│   ├── active/
│   └── history/
└── models/                   # MLモデル用（新規）
    ├── automl_model.pkl
    └── other_model.pkl
```

### 2. App Serviceでの動作確認
- MLページでCSVファイルをアップロード
- Blob Storageからモデルが自動読み込みされる
- 予測結果がダウンロードされる

## トラブルシューティング

### モデルが見つからない場合
1. Blob Storage設定を確認
2. コンテナ名が正しいかチェック
3. アクセス権限を確認

### 認証エラーの場合
1. Managed Identityが有効か確認
2. 適切なロールが割り当てられているかチェック
3. 接続文字列が正しいか確認

## セキュリティ考慮事項

### 本番環境
- ✅ Managed Identity使用
- ✅ 最小権限の原則
- ✅ Private Endpoint設定
- ❌ 接続文字列をコードに埋め込まない

### 開発環境
- 接続文字列をローカルの.envファイルに保存
- .envファイルをGitにコミットしない