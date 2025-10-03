# AI Image Analyzer

画像から撮影地点を推測するためのデモアプリケーションです。ブラウザで画像をアップロードすると、バックエンドが元画像をそのまま Base64 化して OpenRouter のマルチモーダルモデル（`qwen/qwen2.5-vl-72b-instruct:free`）に渡し、人間が現地で観察するような推論プロセスで候補地点を返します。

公開構成は Render を利用し、以下の 2 サービスで動作しています。

| 役割 | Render サービス種別 | 例 | 備考 |
| ---- | ------------------ | --- | ---- |
| バックエンド API | Web Service | `https://image-analyzer-5c3x.onrender.com` | FastAPI + OpenRouter |
| フロントエンド UI | Static Site | `https://image-analyzer-1.onrender.com` | HTML/CSS/JS のみ。API に `fetch` |

> 🎯 **使い方（公開環境）**: フロントURLをブラウザで開き、画像をアップロードすると UI が進捗を表示し、最終的に候補地点・正確性・根拠をカード形式で返します。

---

## リポジトリ構成

```
.
├── backend/
│   └── app.py          # FastAPI アプリケーション（画像処理＆推論）
├── frontend/
│   ├── index.html      # ブラウザ UI（単一ページ）
│   ├── script.js       # UI と API をつなぐロジック
│   └── styles.css      # スタイル定義
├── requirements.txt    # Python 依存パッケージ
├── .env.example        # OpenRouter API キー用テンプレート
└── README.md
```

---

## バックエンド (FastAPI)

### 機能概要

- `POST /analyze` で画像ファイルを受け取り、UUID ベースのタスクを生成
- アップロードされた画像を再エンコードせずそのまま Base64 化し、マルチモーダル入力として利用
- 画像内容とファイル名から組み立てた指示を添えて、人間の現地調査に近い観察→比較→結論の順番で OpenRouter の `qwen/qwen2.5-vl-72b-instruct:free` モデルへ送信
- 初回推論で観察メモと検索クエリ案を生成し、Brave Web Search API でリアルタイム検索。検索クエリやヒットは `notes` に逐次表示し、最終推論にも活用
- 必要に応じて追加質問を生成し、ユーザー回答を受け取って推論を続行
- モデルには「手がかり→候補比較→結論」の 3 ステップで思考させ、看板などの文字情報も読み取って JSON (`location`, `confidence`, `reason`) を返すよう指示
- 進捗はインメモリ辞書 (`tasks`) に `TaskState` として保存し、`GET /status/{task_id}` でポーリング可能
- 推論中の中間メモやリトライは `notes` としてフロントにストリーム表示

### エンドポイント

| メソッド | パス | 説明 |
| --- | --- | --- |
| `GET /health` | ヘルスチェック（常に `{status: "ok"}`） |
| `POST /analyze` | 画像を multipart/form-data で送信。タスク生成後、初期ステータスを返す |
| `GET /status/{task_id}` | 進行中／完了タスクのステータスと結果を返す（JSON は共通フォーマット） |
| `POST /answer/{task_id}` | 追加質問への回答を送信し、推論を再開する |

### 環境変数

- `OPENROUTER_API_KEY`（必須）: OpenRouter の API キー。ローカルでは `.env`（`.env.example` をコピー）に記入し、Render では Environment > Environment Variables で設定します。
- `BRAVE_API_KEY`（任意だが推奨）: Brave Web Search API のサブスクリプションキー。設定すると RAG による補助検索が有効になります。

### 注意事項

- モデル呼び出しは最大 3 回リトライ。一度 JSON 形式にならなければ追加プロンプトで修正を要求
- 進捗バーは疑似的に 80% まで 10 秒で伸び、その後 30 秒かけて 99% まで進行。完了時に 100%
- タスクや画像データはアプリメモリに保持されるため、サーバー再起動で消えます

---

## フロントエンド (静的 HTML/JS)

- `script.js` で `API_BASE_URL` を切り替え。公開時は `https://image-analyzer-5c3x.onrender.com`
- 画像選択時にプレビューを表示しつつ `/analyze` を呼び、返ってきた `task_id` を使って 2 秒ごとに `/status/{task_id}` をポーリング
- 進捗、AI の中間ノート、最終結果カード（場所候補／正確性／推論の根拠）を UI に反映
- 正確性はキャリブレーション済みパーセンテージとカラー付きステータスバーで表示し、低信頼な推論は 40〜60% 程度に抑えて視覚的に区別
- AI から追加情報の質問が届いた場合は、UI 上の回答フォーム経由で `/answer/{task_id}` に送信可能
- `styles.css` で Noto Sans 基調のカードデザイン・バッジ・レスポンシブ対応

---

## ローカル開発手順

1. Python 仮想環境（任意）を作成し、依存関係をインストール
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. `.env.example` をコピーし、OpenRouter API キーを設定
   ```bash
   cp .env.example .env
   # .env を開いて OPENROUTER_API_KEY=sk-or-xxxx を書き込む
   ```

3. バックエンドを起動
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```

4. フロントエンドをローカル HTTP サーバーで配信
   ```bash
   cd frontend
   python -m http.server 8080
   ```

5. ブラウザで `http://localhost:8080` にアクセス → 画像をアップロードして確認

> 補足: ローカルで API を差し替える場合は `frontend/script.js` の `API_BASE_URL` を `http://127.0.0.1:8000` へ変更してください。

---

## Render へのデプロイ手順（参考）

1. GitHub にコミット & push する（`origin` を設定済み）
2. Render ダッシュボードで Web Service を作成
   - リポジトリ: このプロジェクト
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - Environment > Environment Variables に `OPENROUTER_API_KEY` を登録
   - インスタンスは `Free (Starter)` で Auto Suspend 15 分のまま
3. 静的サイトを追加
   - Publish Directory: `frontend`
   - Build Command は空（静的ファイルのみ）
4. `frontend/script.js` の `API_BASE_URL` を Render のバックエンドURLへ設定し、両サービスを再デプロイ
5. バックエンドの `allowed_origins` に静的サイトの URL を追加

---

## API レスポンス例

### `/analyze` 初期レスポンス

```json
{
  "task_id": "c8e3d58a-4a4b-4f7f-8a7f-...",
  "step": "アップロード完了",
  "progress": 0,
  "candidate": null,
  "confidence": null,
  "reason": null,
  "status": "pending",
  "error": null
}
```

### `/status/{task_id}` 完了レスポンス

```json
{
  "task_id": "c8e3d58a-4a4b-4f7f-8a7f-...",
  "step": "解析完了",
  "progress": 100,
  "candidate": "横浜・みなとみらい",
  "confidence": 0.88,
  "reason": "ランドマークタワーと富士山の組み合わせが特徴...",
  "status": "completed",
  "error": null,
  "notes": "推論を開始します...\n[試行1] 手がかりを整理中です。\n..."
}
```

---

## Git / デプロイ運用メモ

- このリポジトリはすでに GitHub (`origin`) と Render に接続済み。ローカルで修正 → `git commit` → `git push` すれば Render が自動再デプロイ
- SSH キーを登録済みのため、今後 push のたびに資格情報入力は不要
- `.env` はローカル専用で、Git にはコミットしないよう `.gitignore` に登録済み
- Render の無料枠では 15 分アクセスがないとスリープ。再開時に数秒かかる点に留意
- 動作確認フェーズでは以下の Git コマンドを順番に実行して GitHub と同期し、Render の自動デプロイをトリガーする:
  1. `git status -sb`
  2. `git add <ファイルパス>`（一括なら `git add .`）
  3. `git commit -m "メッセージ"`
  4. `git push origin master`（ブランチ名が `master` の場合）
- 具体的な例（このREADME更新時）:
  ```bash
  git add backend/app.py requirements.txt README.md
  git status -sb
  git commit -m "Replace OpenCV preprocessing with Pillow and update docs"
  git push origin master
  ```

---

## トラブルシューティング

| 症状 | 原因と対処 |
| ---- | ---------- |
| フロントから 127.0.0.1 にアクセスし `ERR_CONNECTION_REFUSED` | `API_BASE_URL` がローカル用のまま。公開時は Render の URL に変更し、静的サイトを再デプロイ |
| 画像アップロードで 400 | Content-Type が `image/*` ではない or ファイルが空。入力チェックを確認 |
| 推論が失敗 (`status=failed`) | モデルが JSON を返さなかったなど。`notes` に原因（例：「推論に失敗しました: ...」）が記録されるので内容を確認 |
| Render で 404 | `/` ルートはバックエンドに用意していないため正常。静的サイト側 URL にアクセスする |

---

## ライセンス / 利用範囲

このリポジトリはデモ用途のため、API キーや解析結果の取り扱いには十分ご注意ください。必要に応じてログやタスク管理の永続化、OpenRouter の有料モデル利用などを追加実装してください。

---

この README を最新の情報源として更新しておけば、新しいチャットや開発メンバーでもアプリの構成と運用方法を即座に把握できます。
