import asyncio
import base64
import json
import logging
import mimetypes
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI

app = FastAPI(title="Image Analyzer", version="0.1.0")

allowed_origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "http://[::1]:8000",
    "http://[::1]:8080",
    "http://[::]:8000",
    "http://[::]:8080",
    "https://image-analyzer-1.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

OPENAI_MAX_CALLS = 6
MODEL_NAME = "qwen/qwen2.5-vl-72b-instruct:free"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "AI Image Analyzer",
}


@dataclass
class TaskState:
    step: str
    progress: float
    candidate: Optional[str]
    confidence: Optional[float]
    reason: Optional[str]
    notes: Optional[str] = None
    retries: int = 0
    status: str = "pending"
    error: Optional[str] = None
    image_bytes: Optional[bytes] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None


tasks: Dict[str, TaskState] = {}
tasks_lock = asyncio.Lock()
_openai_client: Optional[AsyncOpenAI] = None


def build_payload(task_id: str, state: TaskState) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "step": state.step,
        "progress": state.progress,
        "candidate": state.candidate,
        "confidence": state.confidence,
        "reason": state.reason,
        "status": state.status,
        "error": state.error,
        "notes": state.notes,
    }


async def update_task(task_id: str, **kwargs: Any) -> None:
    async with tasks_lock:
        state = tasks.get(task_id)
        if not state:
            return
        for key, value in kwargs.items():
            setattr(state, key, value)


async def append_note(task_id: str, text: str) -> None:
    async with tasks_lock:
        state = tasks.get(task_id)
        if not state:
            return
        base = state.notes or ""
        separator = "\n" if base else ""
        state.notes = f"{base}{separator}{text.strip()}" if text else base


async def progress_ticker(
    task_id: str,
    target: int,
    duration: float,
    steps: int,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    try:
        async with tasks_lock:
            state = tasks.get(task_id)
        if not state:
            return
        start = state.progress
        if target <= start:
            return

        steps = max(1, steps)
        interval = duration / steps
        increment = (target - start) / steps
        current = float(start)

        for _ in range(steps):
            if stop_event and stop_event.is_set():
                break
            await asyncio.sleep(interval)
            if stop_event and stop_event.is_set():
                break
            current = min(target, current + increment)
            await update_task(task_id, progress=current)
            async with tasks_lock:
                state = tasks.get(task_id)
            if not state or state.status in {"completed", "failed"}:
                break
            if current >= target:
                break
    except asyncio.CancelledError:
        raise


async def manage_progress(task_id: str, stop_event: asyncio.Event) -> None:
    try:
        await progress_ticker(task_id, target=80, duration=10.0, steps=200, stop_event=stop_event)
        if stop_event.is_set():
            return
        await progress_ticker(task_id, target=99, duration=30.0, steps=300, stop_event=stop_event)
        while not stop_event.is_set():
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        raise


async def pop_image(task_id: str) -> Tuple[bytes, Optional[str]]:
    async with tasks_lock:
        state = tasks.get(task_id)
        if not state or not state.image_bytes:
            raise HTTPException(status_code=404, detail="対象タスクの画像データが見つかりません")
        image_bytes = state.image_bytes
        mime_type = state.mime_type
        state.image_bytes = None
        state.mime_type = None
        return image_bytes, mime_type


def ensure_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv(OPENROUTER_KEY_ENV)
        if not api_key:
            raise RuntimeError("OpenRouter APIキーが環境変数に設定されていません")
        _openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
        )
    return _openai_client


def determine_mime_type(mime_type: Optional[str], filename: Optional[str]) -> str:
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed and guessed.startswith("image/"):
            return guessed
    return "image/jpeg"


def build_image_data_uri(image_bytes: bytes, mime_type: Optional[str], filename: Optional[str]) -> str:
    media_type = determine_mime_type(mime_type, filename)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def build_reasoning_prompt(filename: Optional[str]) -> str:
    lines = [
        "以下の写真を現地で観察したつもりで、撮影地点を推測してください。",
        f"元のファイル名: {filename or '不明'}",
        "人間が現地調査で使う手がかり（地形、建物の様式、看板や文字、天候、交通手段、文化的特徴など）を総合的に整理してください。",
        "文字情報は重要な根拠ですが、周囲の風景・構造物・背景の山や海・道路標識など他の要素とも照合して、総合的な判断を行ってください。",
        "確かな根拠と推測的な根拠を区別しながら、最終的な候補地点を決めてください。",
    ]
    return "\n".join(lines)


def normalize_confidence(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num > 1:
        num = num / 100.0
    return max(0.0, min(1.0, num))


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            first = lines[0].strip()
            if first.startswith("```"):
                lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_reasoning_output(content: str) -> Optional[Dict[str, Any]]:
    candidate_text = strip_code_fence(content)
    if "{" in candidate_text and "}" in candidate_text:
        start = candidate_text.find("{")
        end = candidate_text.rfind("}")
        candidate_text = candidate_text[start : end + 1]

    try:
        data = json.loads(candidate_text)
    except json.JSONDecodeError:
        return None

    candidate = data.get("location") or data.get("candidate")
    confidence = normalize_confidence(data.get("confidence"))
    reason = data.get("reason") or data.get("analysis")

    if not candidate or confidence is None or reason is None:
        return None

    return {
        "candidate": candidate,
        "confidence": confidence,
        "reason": reason,
    }


def extract_text_from_message(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text_value: Optional[str] = None
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
            elif hasattr(item, "text"):
                text_value = getattr(item, "text")
            if text_value:
                parts.append(str(text_value))
        return "\n".join(parts).strip()
    return ""


async def run_reasoning_loop(task_id: str, prompt: str, image_data_uri: str) -> Dict[str, Any]:
    client = ensure_openai_client()
    stop_event = asyncio.Event()
    progress_task = asyncio.create_task(manage_progress(task_id, stop_event))
    system_text = (
        "あなたは現地を案内する旅行ガイドAIです。画像を人間が観察するように解析し、場所特定に役立つ手掛かりを整理してください。\n"
        "以下の3段階で推論してください:\n"
        "1. 観察メモ: 建物の構造、景観、気候、地形、道路標識、看板や文字など、人間が現地で目にする要素を箇条書きで整理。確かな観察と推測的な観察を分けてください。\n"
        "2. 候補比較: 2〜3件の候補地を挙げ、各候補と観察メモの照合結果（合致点・不一致点）を比較。\n"
        "3. 最終結論: 最も妥当な候補を選び、決め手となった手掛かりと確信度をまとめる。\n"
        "最終的な出力は JSON で {\"location\": str, \"confidence\": float, \"reason\": str} のみ返してください。"
        "JSON はコードブロック（```）で囲まず、余計なテキストも付けないでください。"
        "location と reason は日本語で、confidence は 0 から 1 の数値にしてください。"
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri,
                    },
                },
            ],
        },
    ]

    result: Optional[Dict[str, Any]] = None

    try:
        for attempt in range(1, OPENAI_MAX_CALLS + 1):
            phases = [
                "手がかりを整理中",
                "候補地を比較中",
                "最終結論を整理中",
            ]

            idx = min(attempt - 1, len(phases) - 1)
            await append_note(task_id, f"[試行{attempt}] {phases[idx]}です。")
            await update_task(
                task_id,
                step=f"{phases[idx]} ({attempt}/{OPENAI_MAX_CALLS})",
                status="processing",
            )
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                )
            except APIError as exc:
                logger.exception("OpenAI API call failed: %s", exc)
                if attempt == OPENAI_MAX_CALLS:
                    raise
                await asyncio.sleep(1.5 * attempt)
                continue

            message_content = response.choices[0].message.content if response.choices else None
            content_text = extract_text_from_message(message_content)

            if not content_text:
                await append_note(task_id, f"[試行{attempt}] モデルから空の応答が返されました。再試行します。")
                messages.append(
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "日本語で指定したJSON形式 (location, confidence, reason) をコードブロックなしで出力してください。",
                            }
                        ],
                    }
                )
                continue

            parsed = parse_reasoning_output(content_text)
            if parsed:
                await append_note(
                    task_id,
                    f"[試行{attempt}] JSON形式の解答を受信しました。候補: {parsed['candidate']}, 確信度: {parsed['confidence']:.2f}",
                )
                result = parsed
                break

            snippet = content_text[:500]
            await append_note(task_id, f"[試行{attempt}] 中間考察:\n{snippet}")
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": content_text}]}
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "前回の内容を踏まえ、日本語で {location, confidence, reason} のJSONをコードブロックなしで返してください。"
                                "confidenceは0から1の範囲です。"
                            ),
                        }
                    ],
                }
            )

        if not result:
            raise RuntimeError("AI応答から有効な結果を取得できませんでした")

        return result
    finally:
        stop_event.set()
        progress_task.cancel()
        with suppress(asyncio.CancelledError):
            await progress_task


async def process_task(task_id: str) -> None:
    try:
        await update_task(task_id, step="解析準備中", progress=10, status="queued")
        image_bytes, mime_type = await pop_image(task_id)
        await update_task(task_id, step="画像を準備中", progress=25, status="processing")

        async with tasks_lock:
            filename = tasks[task_id].filename if task_id in tasks else None

        image_data_uri = build_image_data_uri(image_bytes, mime_type, filename)

        prompt = build_reasoning_prompt(filename)
        await update_task(
            task_id,
            step="AI推論準備中",
            progress=45,
            notes="推論を開始します。現地で観察するように景観や文字の手掛かりを整理しています。",
        )

        result = await run_reasoning_loop(task_id, prompt, image_data_uri)

        await append_note(
            task_id,
            "推論が完了しました。最終結果をUIに反映します。",
        )

        await update_task(
            task_id,
            step="解析完了",
            progress=100,
            status="completed",
            candidate=result["candidate"],
            confidence=result["confidence"],
            reason=result["reason"],
            error=None,
        )
    except RuntimeError as exc:
        logger.error("Task %s failed: %s", task_id, exc)
        await append_note(task_id, f"推論に失敗しました: {exc}")
        await update_task(
            task_id,
            status="failed",
            error=str(exc),
            step="エラーが発生しました",
            progress=100,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while processing task %s", task_id)
        await append_note(task_id, "予期せぬエラーが発生しました。ログを確認してください。")
        await update_task(
            task_id,
            status="failed",
            error="予期せぬエラーが発生しました。再試行してください。",
            step="エラーが発生しました",
            progress=100,
        )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    if not image.filename:
        raise HTTPException(status_code=400, detail="画像ファイルを指定してください")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイル形式のみ受け付けます")

    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="画像データの読み込みに失敗しました")

    task_id = str(uuid.uuid4())
    state = TaskState(
        step="アップロード完了",
        progress=0,
        candidate=None,
        confidence=None,
        reason=None,
        notes=None,
        image_bytes=file_bytes,
        filename=image.filename,
        mime_type=image.content_type,
    )

    async with tasks_lock:
        tasks[task_id] = state

    try:
        asyncio.create_task(process_task(task_id))
    except RuntimeError as exc:
        await update_task(task_id, status="failed", error=str(exc), step="初期化に失敗", progress=100)

    return build_payload(task_id, state)


@app.get("/status/{task_id}")
async def get_status(task_id: str) -> Dict[str, Any]:
    async with tasks_lock:
        state = tasks.get(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="指定されたタスクが見つかりません")
        return build_payload(task_id, state)
