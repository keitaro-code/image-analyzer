import asyncio
import base64
import json
import logging
import mimetypes
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel

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
MODEL_NAME = "qwen/qwen2.5-vl-32b-instruct:free"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "AI Image Analyzer",
}
BRAVE_KEY_ENV = "BRAVE_API_KEY"
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_TIMEOUT = 12.0
BRAVE_MAX_RESULTS = 3


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
    question: Optional[str] = None
    question_context: Optional[str] = None
    awaiting_answer: bool = False
    conversation: Optional[List[Dict[str, Any]]] = None
    planning_snapshot: Optional[Dict[str, Any]] = None
    search_packages_snapshot: Optional[List[Dict[str, Any]]] = None
    image_data_uri: Optional[str] = None


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


@dataclass
class SearchPackage:
    query: str
    rationale: Optional[str]
    results: List[SearchHit]


@dataclass
class PlanningOutput:
    observations_certain: List[str]
    observations_possible: List[str]
    search_queries: List[Dict[str, Optional[str]]]


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
        "question": state.question,
        "question_context": state.question_context,
        "awaiting_answer": state.awaiting_answer,
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


async def brave_web_search(query: str, *, count: int = BRAVE_MAX_RESULTS) -> List[SearchHit]:
    api_key = os.getenv(BRAVE_KEY_ENV)
    if not api_key:
        raise RuntimeError("Brave APIキーが環境変数に設定されていません")

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": count}

    async with httpx.AsyncClient(timeout=BRAVE_TIMEOUT) as client:
        response = await client.get(
            BRAVE_SEARCH_ENDPOINT,
            headers=headers,
            params=params,
        )
        response.raise_for_status()

    data = response.json()
    web_results = data.get("web", {}).get("results", [])
    hits: List[SearchHit] = []
    for item in web_results[:count]:
        title = item.get("title") or "(タイトルなし)"
        url = item.get("url") or item.get("link") or ""
        snippet = item.get("description") or item.get("snippet") or ""
        hits.append(SearchHit(title=title, url=url, snippet=snippet))
    return hits


def build_reasoning_prompt(
    filename: Optional[str],
    planning: PlanningOutput,
    search_packages: List[SearchPackage],
) -> str:
    lines = [
        "以下の写真を現地で観察したつもりで、撮影地点を推測してください。",
        f"元のファイル名: {filename or '不明'}",
        "観察メモとウェブ検索結果を踏まえて、人間のフィールドワークのように根拠を整理してください。",
        "確実な観察と推測的な観察を区別しながら、最終的な候補地点を決めてください。",
    ]

    if planning.observations_certain:
        lines.append("確信できる観察:")
        for item in planning.observations_certain:
            lines.append(f"- {item}")
    if planning.observations_possible:
        lines.append("推測的な観察:")
        for item in planning.observations_possible:
            lines.append(f"- {item}")

    if search_packages:
        lines.append("ウェブ検索から得られた参考情報:")
        for pack in search_packages:
            rationale_text = f"（目的: {pack.rationale}）" if pack.rationale else ""
            lines.append(f"● 検索クエリ: {pack.query}{rationale_text}")
            for hit in pack.results:
                snippet = hit.snippet.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                lines.append(f"  - {hit.title} ({hit.url}): {snippet}")

    lines.append(
        "これらの情報と画像から得られる追加の手掛かりを統合し、最終的な結論を提示してください。"
    )
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


def truncate_for_note(text: str, limit: int = 600) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


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

    status = (data.get("status") or data.get("outcome") or "result").lower()

    if status == "clarification":
        question = data.get("question") or data.get("clarification")
        if not question:
            return None
        context = data.get("context") or data.get("why") or data.get("detail")
        followups = data.get("follow_up") or data.get("followup_questions")
        return {
            "type": "question",
            "question": question,
            "context": context,
            "followups": followups,
        }

    candidate = data.get("location") or data.get("candidate")
    confidence = normalize_confidence(data.get("confidence"))
    reason = data.get("reason") or data.get("analysis")

    if not candidate or confidence is None or reason is None:
        return None

    return {
        "type": "result",
        "candidate": candidate,
        "confidence": confidence,
        "reason": reason,
    }


def parse_planning_output(content: str) -> Optional[PlanningOutput]:
    candidate_text = strip_code_fence(content)
    try:
        data = json.loads(candidate_text)
    except json.JSONDecodeError:
        return None

    observations = data.get("observations") or {}
    certain_raw = observations.get("certain") or observations.get("definite") or observations.get("sure")
    possible_raw = observations.get("possible") or observations.get("speculative") or observations.get("hypotheses")

    def normalize_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    certain = normalize_list(certain_raw)
    possible = normalize_list(possible_raw)

    raw_queries = data.get("search_queries") or data.get("queries") or []
    parsed_queries: List[Dict[str, Optional[str]]] = []
    if isinstance(raw_queries, list):
        for item in raw_queries:
            if isinstance(item, dict):
                query = (item.get("query") or item.get("text") or "").strip()
                if not query:
                    continue
                rationale = item.get("reason") or item.get("purpose") or item.get("goal")
                parsed_queries.append({"query": query, "rationale": (rationale or None)})
            elif isinstance(item, str) and item.strip():
                parsed_queries.append({"query": item.strip(), "rationale": None})

    return PlanningOutput(
        observations_certain=certain,
        observations_possible=possible,
        search_queries=parsed_queries,
    )


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


async def run_planning_phase(
    task_id: str,
    image_data_uri: str,
    filename: Optional[str],
) -> PlanningOutput:
    client = ensure_openai_client()
    system_text = (
        "あなたは現地調査を行う旅行ガイドAIです。画像から観察できる確実な点と推測的な点を整理したうえで、"
        "場所特定に役立つウェブ検索クエリを提案してください。"
        "必ず JSON 形式で {\"observations\":{\"certain\":[],\"possible\":[]},\"search_queries\":[{\"query\":\"...\",\"reason\":\"...\"}]} を返してください。"
        "キー名・配列構造・ダブルクオートを厳守し、null やコメント、説明文を入れないでください。"
        "JSON はコードブロックや前後のテキストで囲まず、単一のオブジェクトのみを出力してください。"
    )
    user_text = (
        "画像を詳細に観察し、人間のフィールドノートのように観察メモをまとめてください。"
        "観察メモは observations.certain と observations.possible に日本語の文字列配列で格納してください。"
        "場所特定のために実行すべきウェブ検索クエリを最大3件 search_queries 配列に {query, reason} 形式で記述してください。"
        "検索が不要と思われる場合は空配列を返してください。"
        "必ず有効な JSON オブジェクトのみを返し、他の文章や説明は出力しないでください。"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "text", "text": f"ファイル名: {filename or '不明'}"},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]

    last_invalid_response: Optional[str] = None

    for attempt in range(1, 4):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
            )
        except APIError as exc:
            logger.exception("Planning call failed: %s", exc)
            if attempt == 3:
                raise
            await asyncio.sleep(1.2 * attempt)
            continue

        message_content = response.choices[0].message.content if response.choices else None
        content_text = extract_text_from_message(message_content)
        parsed = parse_planning_output(content_text)
        if parsed:
            return parsed

        snippet = truncate_for_note(content_text)
        if snippet:
            logger.warning("Planning attempt %s returned non-JSON response: %s", attempt, snippet)
            await append_note(
                task_id,
                "[観察メモ抽出エラー] モデル応答から有効なJSONを解析できませんでした。\n"
                f"{snippet}",
            )
            last_invalid_response = snippet

        messages.append({"role": "assistant", "content": [{"type": "text", "text": content_text}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "出力は {\"observations\":{\"certain\":[],\"possible\":[]},\"search_queries\":[{\"query\":\"...\",\"reason\":\"...\"}]} "
                            "の JSON オブジェクトのみです。ダブルクオートを含む厳密な JSON を生成し、先頭は { で始めて } で終えてください。"
                            "空であっても配列は [] を使用し、説明文やマークダウン、余計な文章を絶対に付けないでください。"
                        ),
                    }
                ],
            }
        )

    error_message = "観察メモと検索クエリの抽出に失敗しました"
    if last_invalid_response:
        error_message += "（詳細は推論ログを参照してください）"
    raise RuntimeError(error_message)


async def run_reasoning_loop(
    task_id: str,
    prompt: str,
    image_data_uri: str,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    client = ensure_openai_client()
    stop_event = asyncio.Event()
    progress_task = asyncio.create_task(manage_progress(task_id, stop_event))
    system_text = (
        "あなたは現地を案内する旅行ガイドAIです。画像を人間が観察するように解析し、場所特定に役立つ手掛かりを整理してください。\n"
        "以下の3段階で推論してください:\n"
        "1. 観察メモ: 建物の構造、景観、気候、地形、道路標識、看板や文字など、人間が現地で目にする要素を箇条書きで整理。確かな観察と推測的な観察を分けてください。\n"
        "2. 候補比較: 2〜3件の候補地を挙げ、各候補と観察メモの照合結果（合致点・不一致点）を比較。\n"
        "3. 最終結論: 最も妥当な候補を選び、決め手となった手掛かりと確信度をまとめる。\n"
        "決定的な手掛かりが欠けている場合だけでなく、観察や検索結果から得られる確証が十分でないと感じたときも clarification を選び、どんな追加情報が必要か日本語で質問を明確に示してください。\n"
        "confidence は 0〜1 の範囲で、根拠が弱い場合は 0.4〜0.6 程度、看板やランドマークなど決定的な証拠が揃ったときのみ 0.75 以上としてください。confidence を 0.65 未満と判断するなら結果ではなく clarification を返してください。\n"
        "最終的な出力は JSON で {status, location, confidence, reason, question, context} を返してください。"
        "status は result もしくは clarification のどちらかです。clarification の場合は question と context を含め、location と confidence は null にしてください。"
        "JSON はコードブロック（```）で囲まず、余計なテキストも付けないでください。"
        "location と reason は日本語で、confidence は 0 から 1 の数値にしてください。"
    )

    if messages is None:
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
                step=phases[idx],
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
                                "text": (
                                    "日本語で指定したJSON形式 (status, location, confidence, reason, question, context) をコードブロックなしで出力してください。"
                                    "confidence が 0.65 未満になりそうな場合は result ではなく clarification を返してください。"
                                ),
                            }
                        ],
                    }
                )
                continue

            parsed = parse_reasoning_output(content_text)
            if parsed:
                assistant_entry = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": content_text}],
                }
                messages.append(assistant_entry)

                if parsed["type"] == "result":
                    await append_note(
                        task_id,
                        f"[試行{attempt}] JSON形式の解答を受信しました。候補: {parsed['candidate']}, 確信度: {parsed['confidence']:.2f}",
                    )
                    return parsed, messages

                if parsed["type"] == "question":
                    await append_note(
                        task_id,
                        "[試行{}] 追加情報が必要と判断されました。質問を提示します。".format(attempt),
                    )
                    return parsed, messages

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
                                "前回の内容を踏まえ、日本語で {status, location, confidence, reason, question, context} のJSONをコードブロックなしで返してください。"
                                "confidence は 0 から 1 の範囲で、0.65 未満になると判断した場合は clarification を選んでください。"
                            ),
                        }
                    ],
                }
            )

        raise RuntimeError("AI応答から有効な結果を取得できませんでした")
    finally:
        stop_event.set()
        progress_task.cancel()
        with suppress(asyncio.CancelledError):
            await progress_task


def format_observations_note(planning: PlanningOutput) -> Optional[str]:
    lines: List[str] = []
    if planning.observations_certain:
        lines.append("確かな観察:")
        lines.extend(f"- {item}" for item in planning.observations_certain)
    if planning.observations_possible:
        if lines:
            lines.append("")
        lines.append("推測的な観察:")
        lines.extend(f"- {item}" for item in planning.observations_possible)
    if not lines:
        return None
    return "\n".join(lines)


def summarize_search_hits(hits: List[SearchHit]) -> str:
    if not hits:
        return "(検索結果なし)"
    lines = []
    for idx, hit in enumerate(hits, start=1):
        snippet = hit.snippet.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        lines.append(f"{idx}. {hit.title} - {snippet}\n   {hit.url}")
    return "\n".join(lines)


def planning_from_snapshot(snapshot: Dict[str, Any]) -> PlanningOutput:
    return PlanningOutput(
        observations_certain=snapshot.get("observations_certain", []) or [],
        observations_possible=snapshot.get("observations_possible", []) or [],
        search_queries=[],
    )


def search_packages_from_snapshot(snapshot: List[Dict[str, Any]]) -> List[SearchPackage]:
    packages: List[SearchPackage] = []
    for item in snapshot:
        query = item.get("query") or ""
        if not query:
            continue
        rationale = item.get("rationale")
        results_data = item.get("results") or []
        hits = [
            SearchHit(
                title=hit.get("title") or "(タイトルなし)",
                url=hit.get("url") or "",
                snippet=hit.get("snippet") or "",
            )
            for hit in results_data
        ]
        packages.append(SearchPackage(query=query, rationale=rationale, results=hits))
    return packages


async def process_task(task_id: str) -> None:
    try:
        await update_task(task_id, step="解析準備中", progress=10, status="queued")
        image_bytes, mime_type = await pop_image(task_id)
        await update_task(task_id, step="分析プランを作成中", progress=25, status="processing")

        async with tasks_lock:
            filename = tasks[task_id].filename if task_id in tasks else None

        image_data_uri = build_image_data_uri(image_bytes, mime_type, filename)
        await update_task(
            task_id,
            image_data_uri=image_data_uri,
            awaiting_answer=False,
            question=None,
            question_context=None,
        )

        planning = await run_planning_phase(task_id, image_data_uri, filename)
        obs_note = format_observations_note(planning)
        if obs_note:
            await append_note(task_id, f"[観察メモ]\n{obs_note}")

        await update_task(task_id, step="ウェブ検索を実行中", progress=45)

        search_packages: List[SearchPackage] = []
        brave_error: Optional[str] = None
        for query_info in planning.search_queries:
            query_text = query_info.get("query") or ""
            if not query_text:
                continue
            rationale = query_info.get("rationale")
            await append_note(
                task_id,
                f"[検索] \"{query_text}\" を実行します。"
                + (f" 目的: {rationale}" if rationale else ""),
            )
            if brave_error:
                await append_note(task_id, f"[検索] スキップ: {brave_error}")
                continue
            try:
                hits = await brave_web_search(query_text)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Brave search failed for task %s: %s", task_id, exc)
                brave_error = str(exc)
                await append_note(task_id, f"[検索] エラー: {brave_error}")
                continue

            await append_note(
                task_id,
                f"[検索結果] \"{query_text}\"\n{summarize_search_hits(hits)}",
            )
            search_packages.append(
                SearchPackage(query=query_text, rationale=rationale, results=hits)
            )

        await update_task(
            task_id,
            step="最終推論準備中",
            progress=60,
            notes="観察メモと検索結果をまとめ、最終推論に進みます。",
        )

        prompt = build_reasoning_prompt(filename, planning, search_packages)
        await update_task(
            task_id,
            step="AI推論中",
            progress=70,
            status="processing",
        )

        result_data, conversation_messages = await run_reasoning_loop(
            task_id,
            prompt,
            image_data_uri,
        )

        snapshot = {
            "observations_certain": planning.observations_certain,
            "observations_possible": planning.observations_possible,
        }
        search_snapshot = [
            {
                "query": pkg.query,
                "rationale": pkg.rationale,
                "results": [
                    {"title": hit.title, "url": hit.url, "snippet": hit.snippet}
                    for hit in pkg.results
                ],
            }
            for pkg in search_packages
        ]

        if result_data["type"] == "question":
            await update_task(
                task_id,
                step="追加情報を待機中",
                progress=80,
                status="awaiting_input",
                question=result_data.get("question"),
                question_context=result_data.get("context"),
                awaiting_answer=True,
                conversation=conversation_messages,
                planning_snapshot=snapshot,
                search_packages_snapshot=search_snapshot,
                image_data_uri=image_data_uri,
            )
            await append_note(
                task_id,
                "追加情報が必要です。画面の質問に回答してください。",
            )
            return

        await append_note(
            task_id,
            "推論が完了しました。最終結果をUIに反映します。",
        )

        await update_task(
            task_id,
            step="解析完了",
            progress=100,
            status="completed",
            candidate=result_data["candidate"],
            confidence=result_data["confidence"],
            reason=result_data["reason"],
            error=None,
            awaiting_answer=False,
            question=None,
            question_context=None,
            conversation=conversation_messages,
            planning_snapshot=snapshot,
            search_packages_snapshot=search_snapshot,
            image_data_uri=image_data_uri,
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
            awaiting_answer=False,
        )


async def continue_with_answer(task_id: str, answer: str) -> None:
    async with tasks_lock:
        state = tasks.get(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="指定されたタスクが見つかりません")
        if not state.awaiting_answer or not state.conversation:
            raise HTTPException(status_code=400, detail="このタスクは追加情報を必要としていません")
        conversation = state.conversation
        planning_snapshot = state.planning_snapshot or {}
        search_snapshot = state.search_packages_snapshot or []
        image_data_uri = state.image_data_uri
        filename = state.filename

    if not image_data_uri:
        raise RuntimeError("画像データが見つからないため推論を再開できません")

    planning = planning_from_snapshot(planning_snapshot)
    search_packages = search_packages_from_snapshot(search_snapshot)
    planning.search_queries = []  # not used in final prompt

    followup_note = f"ユーザー回答: {answer.strip()}"
    await append_note(task_id, followup_note)

    conversation.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"ユーザーからの追加情報: {answer.strip()}",
                }
            ],
        }
    )

    prompt = build_reasoning_prompt(filename, planning, search_packages)
    result_data, updated_conversation = await run_reasoning_loop(
        task_id,
        prompt,
        image_data_uri,
        messages=conversation,
    )

    if result_data["type"] == "question":
        await append_note(task_id, "追加の情報がまだ必要です。続けて回答してください。")
        await update_task(
            task_id,
            step="追加情報を待機中",
            status="awaiting_input",
            question=result_data.get("question"),
            question_context=result_data.get("context"),
            awaiting_answer=True,
            conversation=updated_conversation,
        )
        return

    await append_note(task_id, "推論が完了しました。最終結果をUIに反映します。")

    await update_task(
        task_id,
        step="解析完了",
        progress=100,
        status="completed",
        candidate=result_data["candidate"],
        confidence=result_data["confidence"],
        reason=result_data["reason"],
        error=None,
        awaiting_answer=False,
        question=None,
        question_context=None,
        conversation=updated_conversation,
        planning_snapshot=planning_snapshot,
        search_packages_snapshot=search_snapshot,
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


class AnswerRequest(BaseModel):
    answer: str


@app.post("/answer/{task_id}")
async def submit_answer(task_id: str, payload: AnswerRequest) -> Dict[str, Any]:
    answer = payload.answer.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="回答内容を入力してください")

    async with tasks_lock:
        state = tasks.get(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="指定されたタスクが見つかりません")
        awaiting = state.awaiting_answer

    if not awaiting:
        raise HTTPException(status_code=400, detail="このタスクは追加情報を必要としていません")

    await continue_with_answer(task_id, answer)

    async with tasks_lock:
        state = tasks.get(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="指定されたタスクが見つかりません")
        return build_payload(task_id, state)


@app.get("/status/{task_id}")
async def get_status(task_id: str) -> Dict[str, Any]:
    async with tasks_lock:
        state = tasks.get(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="指定されたタスクが見つかりません")
        return build_payload(task_id, state)
