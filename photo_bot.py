"""
Telegram-бот: присылаешь фото расписания -> получаешь готовый JSON-файл для школьного бота.
Настройки — в файле .env рядом с ботом (или в переменных окружения, например на Render).
Вместе с ботом поднимается маленький веб-сервер (отвечает "ok") — нужен для хостинга на Render.
Зависимости: pip install aiogram aiohttp pillow
"""
import asyncio
import base64
import html
import io
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

import aiohttp
from aiohttp import web
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message

# ---------- настройки из .env ----------
def load_env():
    env = Path(__file__).with_name(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _list(name):
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


load_env()
BOT_TOKEN = os.environ["BOT_TOKEN"]
ADMIN_ID = int(os.environ["ADMIN_ID"])
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACKS = _list("GEMINI_FALLBACKS")
ALLOWED_IDS = [int(x) for x in _list("ALLOWED_IDS") if x.isdigit()]
BATCH_WAIT = int(os.getenv("BATCH_WAIT", "4"))
PROMPT_EXTRA = os.getenv("PROMPT_EXTRA", "").strip()
MAX_RAW_BYTES = 6 * 1024 * 1024           # крупнее — сжимаем (нужен Pillow)
PORT = int(os.getenv("PORT", "10000"))    # Render сам задаёт PORT
SELF_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()  # Render задаёт сам

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("photo_bot")

API = "https://generativelanguage.googleapis.com/v1beta"

PROMPT = """На изображении — школьное расписание уроков на один день: таблицы по классам, в каждой колонке уроки по порядку.
Распознай ВСЕ классы и верни ТОЛЬКО JSON без пояснений и без markdown-обёртки такого вида:
{
  "date": "заголовок/дата расписания как написано, например: Расписание уроков на вторник, 13 октября (пустая строка, если не видно)",
  "classes": {
    "5а": ["Русский язык 202", "История 216", "---", "Математика 316"],
    "11а гуманитарный профиль": ["..."]
  },
  "absent": ["ФИО отсутствующих учителей, если есть такой список, иначе пустой список"],
  "notes": ["замены, объявления и примечания под таблицей, если есть, иначе пустой список"]
}
Правила:
- ключ класса — русские строчные буквы: "5а", "10б", "11а гуманитарный профиль";
- список класса — уроки строго по порядку номеров (1-й элемент = 1-й урок), текст ровно как в ячейке, вместе с кабинетом;
- пустая ячейка = "", прочерк «---» оставь как "---" (так отмечают отсутствие урока в начале/середине дня);
- если у класса есть время начала урока, всё равно оставляй просто строку с предметом;
- строку без номера урока (внеурочка, «ВД ...») добавь отдельным элементом в конец списка этого класса;
- ничего не выдумывай и не исправляй: если текст не читается — оставь "" для этого урока;
- не пропускай классы и не объединяй их."""

LAT2CYR = str.maketrans("abcekmhopxyABCEKMHOPXY", "авсекмнорхуАВСЕКМНОРХУ")
MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8,
          "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}


class BotError(Exception):
    """Ошибка, текст которой показываем пользователю."""


class ModelUnavailable(Exception):
    pass


# ---------- Gemini ----------
def _err_text(data) -> str:
    msg = (data.get("error") or {}).get("message", "") if isinstance(data, dict) else ""
    return msg or str(data)[:300]


async def gemini_request(http, model, parts, gen, note, delays=(0, 5, 15, 30)) -> str:
    url = f"{API}/models/{model}:generateContent"
    payload = {"contents": [{"parts": parts}], "generationConfig": gen}
    last = ""
    for delay in delays:  # повторы при перегрузке / лимите
        if delay:
            await note(f"Gemini занят, жду {delay} с…")
            await asyncio.sleep(delay)
        try:
            async with http.post(url, json=payload, headers={"x-goog-api-key": GEMINI_API_KEY}) as r:
                status, data = r.status, await r.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last = f"сеть: {e}"
            continue
        if status == 200:
            cand = (data.get("candidates") or [{}])[0]
            if cand.get("finishReason") == "MAX_TOKENS":
                raise BotError("Ответ обрезан — слишком много данных на одном фото. Пришли фото частями.")
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
            if not text.strip():
                raise BotError(f"Gemini вернул пустой ответ ({cand.get('finishReason') or data.get('promptFeedback')}).")
            return text
        msg = _err_text(data)
        last = f"HTTP {status}: {msg}"
        log.warning("Gemini %s (модель %s)", last, model)
        if status in (429, 500, 503):
            continue
        if status == 404 or "no longer available" in msg.lower() or "not found" in msg.lower():
            raise ModelUnavailable(msg)
        if "api key" in msg.lower() and ("invalid" in msg.lower() or "not valid" in msg.lower()):
            raise BotError("Ключ Gemini не подходит. Исправь GEMINI_API_KEY в .env.")
        if "location" in msg.lower():
            raise BotError(f"{msg}\nGemini недоступен из региона, где запущен бот — нужен VPN или зарубежный хостинг.")
        raise BotError(f"Ошибка Gemini ({last})")
    if "429" in last:
        raise BotError("Исчерпан бесплатный лимит Gemini. Подожди минуту (или до завтра) и повтори.")
    raise ModelUnavailable(f"перегружена или недоступна ({last})")


def model_chain():
    return [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]


async def ask_gemini(http, images, hint, note) -> str:
    parts = [{"inline_data": {"mime_type": m, "data": base64.b64encode(b).decode()}} for b, m in images]
    extra = "\n\n".join(x for x in (PROMPT_EXTRA, hint) if x)
    parts.append({"text": PROMPT + (f"\n\nДополнительно: {extra}" if extra else "")})
    gen = {"responseMimeType": "application/json", "maxOutputTokens": 32000, "temperature": 0}
    for model in model_chain():
        try:
            return await gemini_request(http, model, parts, gen, note)
        except ModelUnavailable as e:
            log.warning("Модель %s недоступна: %s", model, e)
            await note(f"Модель {model} недоступна, пробую следующую…")
    raise BotError("Ни одна модель не ответила. Проверь GEMINI_MODEL в .env.")


# ---------- разбор и сборка JSON ----------
def extract_json(text: str) -> dict:
    text = re.sub(r"\[cite(?:_start|_end)?\s*:?[^\]]*\]", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("в ответе нет JSON")
    text = text[start:]
    dec = json.JSONDecoder()
    try:
        return dec.raw_decode(text)[0]
    except json.JSONDecodeError:
        return dec.raw_decode(re.sub(r",\s*([}\]])", r"\1", text))[0]


def norm_class(name: str) -> str:
    n = " ".join(str(name).split()).translate(LAT2CYR).lower()
    return re.sub(r"^(\d{1,2})\s+([а-яё])\b", r"\1\2", n)  # "5 а" -> "5а"


def merge(results):
    out = {"date": "", "classes": {}, "absent": [], "notes": []}
    dups = []
    for r in results:
        if not out["date"] and r.get("date"):
            out["date"] = str(r["date"]).strip()
        for name, lessons in (r.get("classes") or {}).items():
            key = norm_class(name)
            cleaned = [("" if x is None else " ".join(str(x).split())) for x in (lessons or [])]
            if key in out["classes"]:
                dups.append(key)
                if len(cleaned) <= len(out["classes"][key]):
                    continue
            out["classes"][key] = cleaned
        for field in ("absent", "notes"):
            for x in r.get(field) or []:
                x = " ".join(str(x).split())
                if x and x not in out[field]:
                    out[field].append(x)
    return out, dups


def date_from_title(title: str):
    m = re.search(r"(\d{1,2})\s+([а-яё]+)", title.lower())
    if m and m.group(2) in MONTHS:
        try:
            return date(date.today().year, MONTHS[m.group(2)], int(m.group(1)))
        except ValueError:
            pass
    return None


def check(sch):
    """Места, которые стоит проверить глазами (ИИ может ошибаться)."""
    warns = []
    if not sch["date"]:
        warns.append("не нашёл дату в заголовке — впиши её в поле \"date\" вручную")
    for cls, ls in sch["classes"].items():
        if not any(x.strip() and x != "---" for x in ls):
            warns.append(f"{cls}: нет ни одного урока")
        for i, x in enumerate(ls, 1):
            if x.strip() and x != "---" and not re.search(r"\d{3}", x):
                warns.append(f"{cls}, {i}-й урок «{x}»: нет номера кабинета")
    return warns


def to_json_text(sch) -> str:
    items = list(sch["classes"].items())
    out = ["{", f'  "date": {json.dumps(sch["date"], ensure_ascii=False)},', '  "classes": {']
    for i, (k, v) in enumerate(items):
        out.append(f'    {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}'
                   + ("," if i < len(items) - 1 else ""))
    out.append("  },")
    out.append(f'  "absent": {json.dumps(sch["absent"], ensure_ascii=False, indent=2).replace(chr(10), chr(10) + "  ")},')
    out.append(f'  "notes": {json.dumps(sch["notes"], ensure_ascii=False, indent=2).replace(chr(10), chr(10) + "  ")}')
    out.append("}")
    text = "\n".join(out) + "\n"
    json.loads(text)  # самопроверка: должен быть корректный JSON
    return text


def prepare_image(raw: bytes, mime: str):
    if len(raw) <= MAX_RAW_BYTES:
        return raw, mime
    try:
        from PIL import Image
    except ImportError:
        raise BotError("Фото больше 6 МБ. Установи Pillow (pip install pillow) или пришли фото поменьше.")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((3000, 3000))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


# ---------- Telegram ----------
router = Router()
LOCK = asyncio.Lock()   # одно распознавание за раз — не упираемся в лимиты бесплатного тарифа
BATCH = {}              # user_id -> {"items": [(file_id, mime)], "hint": str, "task": Task}

HELP = ("📸 <b>Фото расписания → JSON для школьного бота</b>\n\n"
        "Пришли фото (или несколько — альбомом) расписания, и я верну готовый файл <code>.json</code>.\n\n"
        "💡 <b>Советы</b>\n"
        "• Лучше отправлять как <b>файл</b> (скрепка → «Файл»), без сжатия — текст читается точнее.\n"
        "• Если классов много, отправь несколько фото: я распознаю каждое отдельно и склею в один файл.\n"
        "• Подпись к фото — подсказка для ИИ, например: «на фото только 5–7 классы».\n"
        "• ИИ может ошибаться — проверь файл с фото. Я отмечу подозрительные места.\n\n"
        "Дальше: админ-панель школьного бота → «Загрузить файл» → отправь полученный .json.")


def allowed(m: Message) -> bool:
    return bool(m.from_user) and (m.from_user.id == ADMIN_ID or m.from_user.id in ALLOWED_IDS)


@router.message(CommandStart(), F.chat.type == "private")
async def start(m: Message):
    if not allowed(m):
        return await m.answer("⛔ Это приватный бот.")
    await m.answer(HELP + f"\n\n🤖 Модель: <code>{html.escape(GEMINI_MODEL)}</code>")


async def collect(m: Message, file_id: str, mime: str):
    if not allowed(m):
        return await m.answer("⛔ Это приватный бот.")
    b = BATCH.setdefault(m.from_user.id, {"items": [], "hint": "", "task": None})
    b["items"].append((file_id, mime))
    if m.caption:
        b["hint"] = m.caption.strip()
    if b["task"]:
        b["task"].cancel()  # пришло ещё фото — ждём дальше
    b["task"] = asyncio.create_task(run_batch(m, m.from_user.id))


@router.message(F.photo, F.chat.type == "private")
async def on_photo(m: Message):
    await collect(m, m.photo[-1].file_id, "image/jpeg")


@router.message(F.document, F.chat.type == "private")
async def on_document(m: Message):
    mime = m.document.mime_type or ""
    if not mime.startswith("image/"):
        if allowed(m):
            await m.answer("Пришли изображение (фото или файл-картинку).")
        return
    await collect(m, m.document.file_id, mime)


async def run_batch(m: Message, uid: int):
    await asyncio.sleep(BATCH_WAIT)       # если придёт ещё фото, эта задача будет отменена
    b = BATCH.pop(uid, None)
    if not b:
        return
    # дальше отмена уже не должна прерывать обработку
    await asyncio.shield(process(m, b["items"], b["hint"]))


async def process(m: Message, items, hint):
    n = len(items)
    status = await m.answer(f"⏳ Принял {n} фото, в очереди…" if LOCK.locked() else f"⏳ Принял {n} фото, распознаю…")

    async def note(text):
        try:
            await status.edit_text(f"⏳ {text}")
        except Exception:
            pass

    try:
        async with LOCK:
            results = []
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as http:
                for i, (file_id, mime) in enumerate(items, 1):
                    await note(f"Распознаю фото {i} из {n}…")
                    buf = io.BytesIO()
                    await m.bot.download(file_id, destination=buf)
                    raw, mime2 = prepare_image(buf.getvalue(), mime)
                    text = await ask_gemini(http, [(raw, mime2)], hint, note)
                    try:
                        results.append(extract_json(text))
                    except (ValueError, json.JSONDecodeError) as e:
                        raise BotError(f"Не удалось разобрать ответ ИИ на фото {i} ({e}). Попробуй ещё раз.")
        sch, dups = merge(results)
        if not sch["classes"]:
            raise BotError("Не нашёл ни одного класса на фото. Пришли более чёткое фото расписания.")
        text = to_json_text(sch)
    except BotError as e:
        return await status.edit_text(f"❌ {html.escape(str(e))}")
    except Exception:
        log.exception("Ошибка обработки")
        return await status.edit_text("❌ Что-то пошло не так, подробности в логах бота. Попробуй ещё раз.")

    d = date_from_title(sch["date"])
    fname = f"schedule_{d.isoformat()}.json" if d else "schedule.json"
    caption = (f"✅ Готово: классов — {len(sch['classes'])}"
               + (f"\n📅 {html.escape(sch['date'])}" if sch["date"] else ""))
    await m.answer_document(BufferedInputFile(text.encode("utf-8"), filename=fname),
                            caption=caption, parse_mode="HTML")
    await status.delete()

    warns = check(sch)
    if dups:
        warns.append("класс встретился на нескольких фото, взят более полный вариант: " + ", ".join(sorted(set(dups))))
    if warns:
        shown = "\n".join("• " + html.escape(w) for w in warns[:20])
        more = f"\n…и ещё {len(warns) - 20}" if len(warns) > 20 else ""
        await m.answer(f"⚠️ <b>Проверь по фото:</b>\n{shown}{more}", parse_mode="HTML")


# ---------- веб-сервер для Render ----------
async def start_web():
    app = web.Application()

    async def ok(_):
        return web.Response(text="ok")

    app.router.add_get("/", ok)
    app.router.add_get("/health", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()
    log.info("Веб-сервер запущен на порту %s", PORT)


async def keep_alive():
    """Пингуем сами себя, чтобы бесплатный Render не усыплял сервис (раз в 10 минут)."""
    if not SELF_URL:
        return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as http:
        while True:
            await asyncio.sleep(600)
            try:
                async with http.get(SELF_URL):
                    pass
            except Exception as e:
                log.warning("keep-alive: %s", e)


async def main():
    await start_web()
    keep_task = asyncio.create_task(keep_alive())
    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
    dp = Dispatcher()
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    log.info("Бот запущен. Модель: %s", GEMINI_MODEL)
    try:
        await dp.start_polling(bot)
    finally:
        keep_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
