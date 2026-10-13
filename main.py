import os
import json
import re
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

app = FastAPI(title="Table Prism Studio API", version="4.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TYPESAFE_API_KEY = os.environ.get(
    "TYPESAFE_API_KEY",
    "apikey_2189a690f81c4ca54f41850133f0abec3b9c_0dd2b00db7b1e0299eeca8a2ba94be7e2899ac065fd2c0524127c08166c3f28a"
)
TYPESAFE_API_URL = "https://api.typesafe.ai/v1/systemone"

# Список поддерживаемых моделей DuckAI (актуальная линейка)
AVAILABLE_MODELS = [
    {
        "id": "gpt-6-luna",
        "name": "GPT-6 Luna (Best for everyday use)",
        "badge": "Default",
        "desc": "Best for everyday use",
        "provider": "DuckAI/OpenAI"
    },
    {
        "id": "gpt-5.4-mini",
        "name": "GPT-5.4 mini (Solid but hits limits sooner)",
        "badge": "Fast",
        "desc": "Solid but hits limits sooner",
        "provider": "DuckAI/OpenAI"
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5 (Solid but hits limits sooner)",
        "badge": "Anthropic",
        "desc": "Solid but hits limits sooner",
        "provider": "DuckAI/Anthropic"
    },
    {
        "id": "mistral-small-2603",
        "name": "Mistral Small 4",
        "badge": "Mistral",
        "desc": "High efficiency & reasoning",
        "provider": "DuckAI/Mistral"
    },
    {
        "id": "tinfoil/gpt-oss-120b",
        "name": "gpt-oss 120B",
        "badge": "Open-Source",
        "desc": "120B weights powerhouse",
        "provider": "DuckAI/Tinfoil"
    },
    {
        "id": "tinfoil/gemma4-31b",
        "name": "Gemma 4 31B (Beta)",
        "badge": "Beta",
        "desc": "Google Gemma 4 next-gen",
        "provider": "DuckAI/Google"
    }
]

class TableModel(BaseModel):
    id: str
    title: str
    kind: str = "operand" # "operand" | "prism"
    headers: List[str]
    rows: List[List[str]]
    metadata: Optional[Dict[str, Any]] = None

class ConnectionModel(BaseModel):
    id: str
    source_table_id: str
    target_table_id: str

class AnalysisRequest(BaseModel):
    operand_tables: List[TableModel]
    prism_tables: List[TableModel]
    connections: Optional[List[ConnectionModel]] = None

class GenerateTableRequest(BaseModel):
    prompt: str
    kind: str = "operand" # "operand" | "prism"
    model: Optional[str] = "gpt-6-luna"
    max_tokens: Optional[int] = 1000
    unwanted_items: Optional[List[str]] = None
    feedback_reason: Optional[str] = None
    rows_count: int = 5

class GenerateRowRequest(BaseModel):
    table_id: str
    table_title: str
    kind: str = "operand"
    headers: List[str]
    existing_rows: List[List[str]]
    user_instruction: Optional[str] = None
    model: Optional[str] = "gpt-6-luna"
    max_tokens: Optional[int] = 300

class ChatMessage(BaseModel):
    role: str # "system" | "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "gpt-6-luna"
    max_tokens: Optional[int] = 1500

async def call_typesafe(state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {TYPESAFE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jev-latest",
        "state": state,
        "questions": questions
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(TYPESAFE_API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"TypeSafe API error: {resp.text}")
        return resp.json()

def parse_markdown_table_text(text: str) -> Optional[Dict[str, Any]]:
    lines = [l.strip() for l in text.split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]
    if len(lines) >= 2:
        def split_row(row_str):
            cells = row_str.strip()[1:-1].split("|")
            return [c.strip() for c in cells]

        headers = split_row(lines[0])
        row_lines = lines[1:]
        if len(row_lines) > 0 and "---" in row_lines[0]:
            row_lines = row_lines[1:]
        rows = [split_row(rl) for rl in row_lines if "---" not in rl]
        return {"headers": headers, "rows": rows}
    return None

FALLBACK_MODEL_MAP = {
    "gpt-6-luna": "openai",
    "gpt-5.4-mini": "openai-fast",
    "claude-haiku-4-5": "claude",
    "mistral-small-2603": "mistral",
    "tinfoil/gpt-oss-120b": "openai-large",
    "tinfoil/gemma4-31b": "gemma",
}

async def call_ai_llm(messages: List[Dict[str, str]], model: str = "gpt-6-luna", max_tokens: int = 1000) -> str:
    """Try the local OpenAI-compatible service, then retry the remote fallback."""
    request = {"model": model, "messages": messages, "max_tokens": max_tokens}

    def extract_content(response: httpx.Response) -> str:
        response.raise_for_status()
        raw = response.text.strip()
        if not raw:
            return ""
        try:
            data = response.json()
        except ValueError:
            return raw
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content.strip() if isinstance(content, str) else ""
        # Some compatible providers return {"text": "..."} instead of choices.
        text = data.get("text") if isinstance(data, dict) else None
        return text.strip() if isinstance(text, str) else ""

    errors = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(35.0, connect=5.0)) as client:
        # The app's local DuckAI service can be temporarily busy or restarting.
        for attempt in range(2):
            try:
                response = await client.post("http://localhost:3000/v1/chat/completions", json=request)
                content = extract_content(response)
                if content:
                    return content
                errors.append("DuckAI returned an empty response")
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(f"DuckAI: {exc}")
            if attempt == 0:
                await asyncio.sleep(0.6)

        fallback_model = FALLBACK_MODEL_MAP.get(model, "openai-fast")
        fallback_request = {**request, "model": fallback_model}
        for attempt in range(2):
            try:
                response = await client.post("https://text.pollinations.ai/", json=fallback_request)
                content = extract_content(response)
                if content:
                    return content
                errors.append("Fallback provider returned an empty response")
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(f"Fallback provider: {exc}")
            if attempt == 0:
                await asyncio.sleep(0.8)

    print("AI request failed after retries:", "; ".join(errors[-4:]))
    return ""

@app.get("/api/models")
async def get_models():
    """Возвращает список моделей для выбора в интерфейсе."""
    return {"models": AVAILABLE_MODELS}

# ГЕНЕРАТОР ТАБЛИЦ С ВЫБОРОМ МОДЕЛЕЙ И ТОКЕНОВ
@app.post("/api/generate-table")
async def generate_table(req: GenerateTableRequest):
    system_instruction = (
        "You are an expert table synthesizer. "
        "Create a clean, highly structured Markdown table according to user prompt. "
        "Return ONLY the markdown table. Do not include introductory text, explanations, or quotes."
    )

    user_query = req.prompt
    if req.kind == "prism":
        user_query += " (Формат таблицы: Вектор-критерий | Направленность градации (более выгодно/сильнее/стабильнее) | Условие доминирования)"
    else:
        user_query += " (Формат: Вариант/Решение | Категория | Механизм/Принцип)"

    feedback_clause = ""
    if req.unwanted_items and len(req.unwanted_items) > 0:
        feedback_clause += f"\nИсключи и замени следующие неподходящие строки: {json.dumps(req.unwanted_items, ensure_ascii=False)}.\n"
    if req.feedback_reason:
        feedback_clause += f"Причина почему они не подошли (критика пользователя): {req.feedback_reason}. Учти это и сделай новые строки принципиально лучше и точнее!"

    full_prompt = f"Тема таблицы: {user_query}\n{feedback_clause}\nКоличество строк: {req.rows_count}."

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": full_prompt}
    ]

    content = await call_ai_llm(messages, model=req.model or "gpt-6-luna", max_tokens=req.max_tokens or 1000)
    table_data = parse_markdown_table_text(content) if content else None

    # Fallback генератор если сеть недоступна
    if not table_data:
        if req.kind == "prism":
            table_data = {
                "headers": ["Вектор Призмы", "Градация", "Критерий Доминирования"],
                "rows": [
                    [f"Более выгодно по {req.prompt}", "Максимум LTV", "Генерирует наибольшую отдачу"],
                    [f"Более стабильно по {req.prompt}", "Низкая волатильность", "Устойчив к внешним сдвигам"],
                    [f"Более масштабируемо по {req.prompt}", "Высокая пропускная способность", "Минимальные удельные издержки"]
                ]
            }
        else:
            table_data = {
                "headers": ["Решение", "Категория", "Принцип работы"],
                "rows": [
                    [f"Решение 1: {req.prompt}", "Базовый класс", "Обеспечивает фундамент"],
                    [f"Решение 2: {req.prompt}", "Инновационный класс", "Максимизирует эффективность"],
                    [f"Решение 3: {req.prompt}", "Оптимизационный класс", "Устраняет узкие места"]
                ]
            }

    table_id = "gen_" + str(int(os.times().elapsed * 1000))
    return {
        "table": {
            "id": table_id,
            "title": f"{'💎 Призма' if req.kind == 'prism' else '📊 Операнд'}: {req.prompt[:35]}",
            "kind": req.kind,
            "headers": table_data["headers"],
            "rows": table_data["rows"]
        },
        "model_used": req.model,
        "max_tokens": req.max_tokens,
        "prompt": req.prompt,
        "feedback_applied": bool(req.feedback_reason or req.unwanted_items)
    }

# ПОСТРОЧНАЯ ГЕНЕРАЦИЯ В СУЩЕСТВУЮЩУЮ ТАБЛИЦУ
@app.post("/api/generate-row")
async def generate_row(req: GenerateRowRequest):
    """Генерирует новую строку в таблицу с учетом ее заголовков, контекста и пользовательской инструкции."""
    headers_str = ", ".join(req.headers)
    existing_sample = "; ".join([" | ".join(r) for r in req.existing_rows[-3:]]) if req.existing_rows else "Пока нет строк"

    system_instruction = (
        "You are an expert data generator. "
        "Generate exactly ONE new row that perfectly fits the given table schema. "
        "Return strictly a JSON array of strings corresponding to the headers. Example: [\"val1\", \"val2\", \"val3\"]. "
        "Do not output markdown quotes, text or explanations."
    )

    prompt = (
        f"Таблица: '{req.table_title}' (Тип: {req.kind})\n"
        f"Заголовки колонок: [{headers_str}]\n"
        f"Существующие строки контекста: {existing_sample}\n"
    )
    if req.user_instruction:
        prompt += f"Специфическое пожелание пользователя для новой строки: {req.user_instruction}\n"
    prompt += f"Создай ровно одну новую качественную и уникальную строку в виде JSON-массива из {len(req.headers)} элементов."

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]

    content = await call_ai_llm(messages, model=req.model or "gpt-6-luna", max_tokens=req.max_tokens or 300)
    
    # Парсим JSON-массив строки
    row = None
    if content:
        try:
            # Ищем массив [...]
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    row = [str(x) for x in parsed]
        except Exception:
            pass

    # Если LLM вернул в формате | a | b |
    if not row and content and "|" in content:
        parts = [p.strip() for p in content.strip().split("|") if p.strip()]
        if len(parts) >= len(req.headers):
            row = parts[:len(req.headers)]

    # Fallback если не удалось распарсить
    if not row or len(row) != len(req.headers):
        instruction_label = f" ({req.user_instruction[:20]})" if req.user_instruction else ""
        row = [f"Новый пункт {len(req.existing_rows) + 1}{instruction_label}"] + [f"Параметр {i+1}" for i in range(len(req.headers) - 1)]

    return {
        "row": row,
        "model_used": req.model,
        "max_tokens": req.max_tokens
    }

@app.post("/api/chat")
async def chat_interaction(req: ChatRequest):
    """Интерактивный диалоговый чат с DuckAI для обсуждения, итеративного уточнения и генерации таблиц."""
    system_prompt = (
        "Ты — высококлассный аналитический ассистент и архитектор данных в Table Prism Studio. "
        "Твоя задача — вести живой, продуктивный разговор с пользователем: обсуждать нюансы его предметной области, "
        "помогать кристаллизовать критерии сравнения (Таблицы-Призмы) и формулировать варианты решений/стратегий (Таблицы-Операнды). "
        "Когда пользователь просит сгенерировать таблицу (или когда разговор готов к фиксации структуры), "
        "создавай чистые, профессиональные Markdown таблицы вида:\n\n"
        "| Колонка 1 | Колонка 2 | Колонка 3 |\n"
        "| --- | --- | --- |\n"
        "| Значение 1 | Значение 2 | Значение 3 |\n\n"
        "Каждую таблицу обязательно предваряй четким заголовком, например: '### 📊 Операнд: Название' или '### 💎 Призма: Название'. "
        "Пользователь прямо в интерфейсе чата может нажать кнопку для выбора отдельных строк или переноса таблицы целиком на холст."
    )

    formatted_messages = []
    has_system = any(m.role == "system" for m in req.messages)
    if not has_system:
        formatted_messages.append({"role": "system", "content": system_prompt})

    for m in req.messages:
        formatted_messages.append({"role": m.role, "content": m.content})

    reply = await call_ai_llm(formatted_messages, model=req.model or "gpt-6-luna", max_tokens=req.max_tokens or 1500)

    if not reply:
        raise HTTPException(
            status_code=502,
            detail="Модель не вернула ответ после повторных попыток. Повторите запрос или выберите другую модель."
        )

    return {
        "reply": reply,
        "model_used": req.model
    }

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model": "jev-latest"}

@app.post("/api/analyze")
async def analyze_tables(req: AnalysisRequest):
    if not req.operand_tables:
        raise HTTPException(status_code=400, detail="Требуется хотя бы одна таблица-операнд")
    if not req.prism_tables:
        raise HTTPException(status_code=400, detail="Требуется хотя бы одна таблица-призма")

    # Индексы в строках результата всегда относятся к исходной таблице.
    # Идентификаторы таблиц должны быть уникальны: иначе стрелки и бейджи
    # невозможно однозначно привязать к строкам на холсте.
    table_ids = [t.id for t in req.operand_tables + req.prism_tables]
    if len(table_ids) != len(set(table_ids)):
        raise HTTPException(status_code=400, detail="ID таблиц должны быть уникальными")

    # Собираем критерии со ВСЕХ таблиц-призм
    prism_criteria = []
    for pr_t in req.prism_tables:
        for r_idx, r in enumerate(pr_t.rows):
            rule_str = " | ".join([str(c) for c in r if str(c).strip()])
            if rule_str:
                prism_criteria.append({
                    "id": f"{pr_t.id}_c{r_idx}",
                    "prism_table_id": pr_t.id,
                    "prism_table_title": pr_t.title,
                    "index": r_idx,
                    "name": str(r[0]).strip() if len(r) > 0 and str(r[0]).strip() else f"Критерий #{r_idx+1}",
                    "rule": rule_str
                })

    if not prism_criteria:
        raise HTTPException(status_code=400, detail="В таблицах-призмах нет строк критериев")

    # Собираем элементы со всех таблиц-операндов
    items = []
    for t in req.operand_tables:
        for r_idx, r in enumerate(t.rows):
            s = " | ".join([str(c) for c in r if str(c).strip()])
            if s:
                items.append({
                    "id": f"{t.id}_r{r_idx}",
                    "table_id": t.id,
                    "table_title": t.title,
                    "row_index": r_idx,
                    "row_data": r,
                    "content": s
                })

    if not items:
        raise HTTPException(status_code=400, detail="В таблицах-операндах нет строк данных")

    row_to_row_lines = []
    branch_results = []
    item_scores_map = {it["id"]: [] for it in items}
    evaluation_sources = {it["id"]: [] for it in items}

    def normalized_score(answer):
        """Return only an explicit finite score in the declared 0..3 scale."""
        if isinstance(answer, dict):
            answer = answer.get("score")
        try:
            value = float(answer)
        except (TypeError, ValueError):
            return None
        if not __import__("math").isfinite(value) or not 0 <= value <= 3:
            return None
        return value

    def lexical_fallback(criterion, item):
        # Low-confidence fallback; never presented as an AI evaluation.
        words = lambda value: {w for w in re.findall(r"[\w-]+", value.casefold()) if len(w) > 2}
        left, right = words(criterion), words(item)
        return round(3 * len(left & right) / len(left | right), 2) if left and right else 0.0

    for crit in prism_criteria:
        scored_candidates = []
        # Keep requests bounded while scoring every operand row (not just the first 20).
        for offset in range(0, len(items), 20):
            batch = items[offset:offset + 20]
            questions = {
                f"score_{idx}": {
                    "type": "score",
                    "instructions": {
                        "item": it["content"],
                        "prism_criterion": crit["rule"],
                        "question": f"Оцените силу соответствия операнда критерию призмы '{crit['name']}'."
                    },
                    "criteria": [
                        "0 — соответствия нет", "1 — слабое соответствие",
                        "2 — умеренное соответствие", "3 — сильное соответствие"
                    ]
                }
                for idx, it in enumerate(batch)
            }
            answers = {}
            try:
                result = await call_typesafe(
                    state={"criterion": crit["rule"], "items_count": len(batch)},
                    questions=questions
                )
                answers = result.get("answers", {}) if isinstance(result, dict) else {}
            except Exception:
                pass

            for idx, it in enumerate(batch):
                raw = answers.get(f"score_{idx}", {}) if isinstance(answers, dict) else {}
                score = normalized_score(raw)
                source = "typesafe"
                if score is None:
                    score = lexical_fallback(crit["rule"], it["content"])
                    source = "lexical_fallback"
                item_scores_map[it["id"]].append(score)
                evaluation_sources[it["id"]].append(source)
                scored_candidates.append((score, it, source))

        # The visible arrow is derived from the same validated scores as the ranking.
        best_score = max(candidate[0] for candidate in scored_candidates)
        best_score, matched_item, match_source = next(
            candidate for candidate in scored_candidates if candidate[0] == best_score
        )
        row_to_row_lines.append({
            "line_id": f"line_{crit['id']}_{matched_item['id']}",
            "prism_table_id": crit["prism_table_id"],
            "prism_row_index": crit["index"],
            "prism_criterion_name": crit["name"],
            "operand_table_id": matched_item["table_id"],
            "operand_table_title": matched_item["table_title"],
            "operand_row_index": matched_item["row_index"],
            "operand_content": matched_item["content"],
            "confidence": 0.35 if match_source == "lexical_fallback" else None,
            "score": best_score,
            "score_source": match_source
        })
        branch_results.append({
            "criterion_id": crit["id"],
            "criterion_index": crit["index"],
            "criterion_name": crit["name"],
            "criterion_rule": crit["rule"],
            "branch_leader": matched_item
        })

    inter_operand_lines = []
    if len(req.operand_tables) >= 2:
        t_a = req.operand_tables[0]
        t_b = req.operand_tables[1]
        t_a_items = [it for it in items if it["table_id"] == t_a.id]
        t_b_items = [it for it in items if it["table_id"] == t_b.id]

        if t_a_items and t_b_items:
            cross_q = {}
            for idx_a, it_a in enumerate(t_a_items[:8]):
                b_options = {f"b_{j}": it_b["content"] for j, it_b in enumerate(t_b_items[:8])}
                cross_q[f"cross_{idx_a}"] = {
                    "type": "choice",
                    "instructions": {
                        "item_from_A": it_a["content"],
                        "candidates_from_B": [it_b["content"] for it_b in t_b_items[:8]],
                        "question": "Какая строка из Таблицы Б образует наиболее сильную связку (синтез или парное соотношение) со строкой из Таблицы А?"
                    },
                    "criteria": b_options
                }

            cross_ans = {}
            try:
                res_cross = await call_typesafe(state={"mode": "cross_operands"}, questions=cross_q)
                cross_ans = res_cross.get("answers", {}) if isinstance(res_cross, dict) else {}
            except Exception:
                # No inferred pair is emitted when the model is unavailable.
                pass

            for idx_a, it_a in enumerate(t_a_items[:8]):
                ans = cross_ans.get(f"cross_{idx_a}", {})
                choice = ans.get("choice", "") if isinstance(ans, dict) else ""
                if isinstance(choice, str) and re.fullmatch(r"b_\d+", choice):
                    j = int(choice[2:])
                    if j < len(t_b_items[:8]):
                        it_b = t_b_items[j]
                        confidence = ans.get("confidence")
                        try:
                            confidence = float(confidence)
                            if not __import__("math").isfinite(confidence) or not 0 <= confidence <= 1:
                                confidence = None
                        except (TypeError, ValueError):
                            confidence = None
                        inter_operand_lines.append({
                                "line_id": f"cross_{it_a['id']}_{it_b['id']}",
                                "source_table_id": it_a["table_id"],
                                "source_row_index": it_a["row_index"],
                                "source_content": it_a["content"],
                                "target_table_id": it_b["table_id"],
                                "target_row_index": it_b["row_index"],
                                "target_content": it_b["content"],
                                "relation_type": "synergy_pair",
                                "confidence": round(confidence, 2) if confidence is not None else None,
                                "scope": "first_8_rows_per_table"
                        })

    ranked_items = []
    for it in items:
        scores = item_scores_map[it["id"]]
        avg_score = round(sum(scores) / len(scores), 2)
        ranked_items.append({
            **it,
            "integral_score": avg_score,
            "evaluated_criteria": len(scores),
            "score_sources": sorted(set(evaluation_sources[it["id"]]))
        })
    ranked_items.sort(key=lambda x: (-x["integral_score"], x["table_id"], x["row_index"]))

    # "Доминант" means tied for the actual highest aggregate score; avoid arbitrary
    # median splits that label weak rows as leaders even when every score is low.
    highest_score = ranked_items[0]["integral_score"]
    top_leaders = [it for it in ranked_items if highest_score > 0 and it["integral_score"] == highest_score]
    top_ids = {it["id"] for it in top_leaders}
    subordinate_items = [it for it in ranked_items if it["id"] not in top_ids]

    # Структурируем оценки по таблицам для красивого бейджинга на холсте
    evaluations_by_table = {}
    for it in ranked_items:
        t_id = it["table_id"]
        r_idx = it["row_index"]
        if t_id not in evaluations_by_table:
            evaluations_by_table[t_id] = {}
        evaluations_by_table[t_id][r_idx] = {
            "integral_score": it["integral_score"],
            "is_top": it["id"] in top_ids
        }

    sample_headers = ["Источник таблицы"] + list(dict.fromkeys(
        header for table in req.operand_tables for header in table.headers
    )) + ["Интегральный Score", "Критериев учтено", "Категория Доминирования"]
    header_indexes = {header: i for i, header in enumerate(sample_headers)}
    def normalized_row(item, label):
        source_table = next(table for table in req.operand_tables if table.id == item["table_id"])
        values = [""] * (len(sample_headers) - 4)
        for index, header in enumerate(source_table.headers):
            target_index = header_indexes.get(header)
            if target_index is not None and index < len(item["row_data"]):
                values[target_index - 1] = item["row_data"][index]
        return [item["table_title"]] + values + [
            f"{item['integral_score']}/3", str(item["evaluated_criteria"]), label
        ]
    normalized_top_table = {
        "id": "norm_top_" + uuid.uuid4().hex,
        "title": f"🏆 ТОП-Лидеры (Доминантная позиция)",
        "kind": "operand",
        "headers": sample_headers,
        "rows": [normalized_row(it, "Доминант") for it in top_leaders]
    }
    normalized_sub_table = {
        "id": "norm_sub_" + uuid.uuid4().hex,
        "title": f"⚖️ Субординированные (Зона оптимизации)",
        "kind": "operand",
        "headers": sample_headers,
        "rows": [normalized_row(it, "Субординат") for it in subordinate_items]
    }

    return {
        "mode": "cross_operands" if len(req.operand_tables) >= 2 else "internal_operand",
        "row_to_row_lines": row_to_row_lines,
        "inter_operand_lines": inter_operand_lines,
        "branches": branch_results,
        "ranked_items": ranked_items,
        "evaluations_by_table": evaluations_by_table,
        "evaluation_complete": all(len(item_scores_map[it["id"]]) == len(prism_criteria) for it in items),
        "top_leaders": top_leaders,
        "subordinate_items": subordinate_items,
        "normalized_top_table": normalized_top_table,
        "normalized_sub_table": normalized_sub_table,
        "absolute_dominant": ranked_items[0] if top_leaders else None
    }

os.makedirs("/home/madiyar/table_prism_studio/static", exist_ok=True)
app.mount("/", StaticFiles(directory="/home/madiyar/table_prism_studio/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8400, reload=True)
