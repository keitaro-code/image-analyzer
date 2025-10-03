import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional

from PIL import Image, UnidentifiedImageError
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

MAX_IMAGE_SIZE = 512
OPENAI_MAX_CALLS = 3
MODEL_NAME = "x-ai/grok-4-fast:free"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "AI Image Analyzer",
}

try:
    RESAMPLE_MODE = Image.Resampling.LANCZOS  # Pillow>=9
except AttributeError:  # pragma: no cover - Pillow<9 fallback
    RESAMPLE_MODE = Image.LANCZOS


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


async def pop_image(task_id: str) -> bytes:
    async with tasks_lock:
        state = tasks.get(task_id)
        if not state or not state.image_bytes:
            raise HTTPException(status_code=404, detail="対象タスクの画像データが見つかりません")
        image_bytes = state.image_bytes
        state.image_bytes = None
        return image_bytes


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


def prepare_image_for_model(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            rgb_img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("画像の読み込みに失敗しました") from exc

    rgb_img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), RESAMPLE_MODE)

    buffer = BytesIO()
    rgb_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_reasoning_prompt(filename: Optional[str]) -> str:
    lines = [
        "以下の画像について、撮影場所や特徴的なランドマークを推定してください。",
        f"元のファイル名: {filename or '不明'}",
        "画像内の文字（縦書き・斜め文字・日本語を含む）も読み取って、推定根拠に活用してください。",
        "看板・建物・風景などから得られる手がかりを箇条書きで整理したうえで最終候補を決めてください。",
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


def parse_reasoning_output(content: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(content)
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


async def run_reasoning_loop(task_id: str, prompt: str, image_b64: str) -> Dict[str, Any]:
    client = ensure_openai_client()
    stop_event = asyncio.Event()
    progress_task = asyncio.create_task(manage_progress(task_id, stop_event))
    system_text = (
        "あなたは日本語で思考過程を行う旅行ガイドAIです。画像から読み取れる視覚的な手がかりや写っている文字情報を活用して推論してください。\n"
        "以下の3段階で推論してください:\n"
        "1. 手がかり抽出: 建物・風景・標識・看板・写っている文字（縦書きや斜め文字を含む）から場所特定に役立つ情報を箇条書きで整理。\n"
        "2. 候補比較: 2〜3件の候補地を挙げ、それぞれ手がかりとの一致度・不足点を比較。\n"
        "3. 最終結論: 最も確からしい場所を選び、理由と確信度をまとめる。\n"
        "最終的な出力は JSON で {\"location\": str, \"confidence\": float, \"reason\": str} のみ返してください。"
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
                        "url": f"data:image/jpeg;base64,{image_b64}",
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
                                "text": "日本語で指定したJSON形式 (location, confidence, reason) を出力してください。",
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
                                "前回の内容を踏まえ、日本語で {location, confidence, reason} のJSONを返してください。"
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
        image_bytes = await pop_image(task_id)
        await update_task(task_id, step="画像を正規化中", progress=25, status="processing")

        loop = asyncio.get_running_loop()
        image_b64 = await loop.run_in_executor(None, prepare_image_for_model, image_bytes)

        async with tasks_lock:
            filename = tasks[task_id].filename if task_id in tasks else None

        prompt = build_reasoning_prompt(filename)
        await update_task(
            task_id,
            step="Grok推論準備中",
            progress=45,
            notes="推論を開始します。画像の視覚情報と文字情報を読み取っています。",
        )

        result = await run_reasoning_loop(task_id, prompt, image_b64)

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
