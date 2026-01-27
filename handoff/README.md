# AI Image Analyzer - Handoff

画像から撮影地点を推測するデモアプリの引き渡し用パッケージです。FastAPI バックエンドと静的フロントエンド（HTML/CSS/JS）のコピーを同梱しています（元リポジトリは未変更）。

## 何を実行するか（最小手順）
- 依存インストール: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- 環境変数: このディレクトリ直下の `.env.example` を `.env` にコピーし、`OPENROUTER_API_KEY` を設定（必須）。`BRAVE_API_KEY` は任意。
- バックエンド起動: `uvicorn backend.app:app --reload --port 8000`
- フロントエンド配信: `cd frontend && python -m http.server 8080`
- 接続先: 受領者側のバックエンドURL（例: `http://127.0.0.1:8000` や自前デプロイURL）を `frontend/script.js` の `API_BASE_URL` に設定する前提。公開API例は使わない。

## 実行対象ファイルと役割
- `backend/app.py`（コピー）: FastAPI アプリ。`/analyze`, `/status/{task_id}`, `/answer/{task_id}`, `/health`。OpenRouter マルチモーダルモデル `google/gemini-3-pro-preview` を利用。
- `frontend/index.html` / `frontend/script.js` / `frontend/styles.css`（コピー）: UI一式。`API_BASE_URL` で接続先を切替。
- `requirements.txt`（コピー）: Python 依存ライブラリ。
- `.env.example`（コピー）: 秘密値なしテンプレート。

## API の概要
- `GET /health`: ヘルスチェック（`{"status":"ok"}`）。
- `POST /analyze`: multipart で画像を受け取り、タスクIDと初期ステータスを返す。
- `GET /status/{task_id}`: 進捗・推論ログ・結果を返す。`status` は `pending` / `processing` / `awaiting_input` / `completed` / `failed`。
- `POST /answer/{task_id}`: 追加質問への回答（テキストと最大3枚の画像）を送信し、推論を再開。

## 環境変数
- `OPENROUTER_API_KEY`（必須）: OpenRouter の API キー。未設定だと起動時にエラー。
- `BRAVE_API_KEY`（任意）: Brave Web Search API キー。設定すると検索結果を推論に組み込みます。

## 運用上の注意
- タスク状態と画像はプロセスメモリにのみ保持。プロセス再起動で消えます。
- Brave 検索はレート制限時にスキップし、ログに理由を記録します。
- CORS は localhost/127.0.0.1 (8000/8080) と公開フロント `https://image-analyzer-1.onrender.com` を許可済み。別フロントを使う場合は CORS 設定を追加してください。
