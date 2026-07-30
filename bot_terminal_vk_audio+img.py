import os
import re
import sys
import json
import time
import logging
from pathlib import Path

import requests  # уже есть в зависимостях vk_api; качаем им вложения с CDN ВК
import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.utils import get_random_id
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat
from pydantic import BaseModel, Field, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем переменные из .env файла
load_dotenv()

# --- КОНФИГУРАЦИЯ ---
VK_TOKEN = os.getenv('VK_API')
GROUP_ID = int(os.getenv('GROUP_ID'))
GIGACHAT_API_KEY = os.getenv('GIGACHAT_CREDENTIALS') or os.getenv('GIGACHAT_API_KEY')

# ВК требует переводить API-интеграции на .ru (дедлайн был 30.09.2025).
# Верни 'https://api.vk.com/', если .ru вдруг заупрямится.
VK_API_HOST = 'https://api.vk.ru/'

# Настройки GigaChat
GIGACHAT_MODEL = "GigaChat-2"  # Текст (решение). Freemium: GigaChat-2 (Lite), -2-Pro, -2-Max, GigaChat-3-Ultra
# Картинки и аудио на ВХОДЕ понимают только GigaChat-2-Pro и GigaChat-2-Max.
# GigaChat-3-Ultra в API — текстовая (в карточке модели мультимодального входа нет).
# Во freemium на Pro отдельный пакет 40 млн токенов — распознавание вложений делаем им.
RECOGNITION_MODEL = "GigaChat-2-Pro"
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"  # Для физических лиц

MAX_INPUT_LEN = 500     # символов на пользовательский ввод
MAX_HISTORY = 10        # последних сообщений диалога (помимо system)
TOKEN_ALERT = 1000      # алерт в stderr при превышении токенов на ответ
VK_MSG_LIMIT = 4000     # у ВК потолок ~4096 символов на сообщение
MAX_VOICE_SEC = 90      # голосовые длиннее не разбираем — экономим токены
MAX_IMAGE_MB = 15       # лимит GigaChat API на одно изображение
MAX_AUDIO_MB = 35       # лимит GigaChat API на один аудиофайл
DOWNLOAD_TIMEOUT = 20   # секунд на скачивание вложения с CDN ВК

CORPUS_PATH = Path(__file__).resolve().parent / 'corpus.txt'
LOG_PATH = Path(__file__).resolve().parent / 'bot.log'

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8',
)

# ============================================================
# Кубик 1. Корпус, TF-IDF индекс, retrieve
# ============================================================

def load_corpus(path=CORPUS_PATH):
    """Читаем corpus.txt — куски разделены пустой строкой."""
    text = Path(path).read_text(encoding='utf-8')
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        raise RuntimeError('corpus.txt пустой — проверь файл')
    return paragraphs

CORPUS = load_corpus()
VECTORIZER = TfidfVectorizer()
MATRIX = VECTORIZER.fit_transform(CORPUS)

def normalize(text):
    """Нижний регистр + без пунктуации."""
    return re.sub(r'[^\w\s]', '', text.lower())

def retrieve(query, k=2):
    """TF-IDF + cosine similarity. Возвращает k самых релевантных параграфов."""
    q_vec = VECTORIZER.transform([query])
    sims = cosine_similarity(q_vec, MATRIX).ravel()
    top_idx = sims.argsort()[-k:][::-1]
    return [CORPUS[i] for i in top_idx]

# ============================================================
# Кубик 2. System-роль учитель + RAG-контекст
# ============================================================

system_text = (
    'Ты учитель математики и физики. '
    'Отвечай СТРОГО в JSON-формате: '
    '{"given": "...", "find": "...", "steps": [...], "answer": "..."}. '
    'Никакого текста до и после, никаких тройных кавычек. '
    'Используй ТОЛЬКО материалы ниже. Не используй свои знания. Не сочиняй. '
    'Если ответа в материалах нет — верни ровно такой JSON: '
    '{"given": "-", "find": "-", "steps": ["в учебнике нет"], "answer": "в учебнике нет"}.\n\n'
)

def build_system(user_text):
    """system_text + top-2 куска корпуса. Конкатенация, а не .format():
    в шаблоне есть фигурные скобки JSON, .format() на них падает."""
    context = '\n\n'.join(retrieve(normalize(user_text), k=2))
    return system_text + 'Материалы:\n' + context

# ============================================================
# Кубик 3. Few-shot: 2 пары в JSON
# ============================================================

FEW_SHOT = [
    {
        'role': 'user',
        'content': 'Реши уравнение x + 5 = 12',
    },
    {
        'role': 'assistant',
        'content': json.dumps({
            'given': 'x + 5 = 12',
            'find': 'x',
            'steps': ['x = 12 - 5', 'x = 7'],
            'answer': 'x = 7',
        }, ensure_ascii=False),
    },
    {
        'role': 'user',
        'content': 'Реши уравнение 3x = 15',
    },
    {
        'role': 'assistant',
        'content': json.dumps({
            'given': '3x = 15',
            'find': 'x',
            'steps': ['x = 15 / 3', 'x = 5'],
            'answer': 'x = 5',
        }, ensure_ascii=False),
    },
]

# ============================================================
# Кубик 4. Solution pydantic + regex
# ============================================================

class Solution(BaseModel):
    given: str = Field(..., description='что дано в задаче')
    find: str = Field(..., description='что найти')
    steps: list[str] = Field(..., min_length=1, description='шаги решения')
    answer: str = Field(..., description='итоговый ответ')

def parse_solution(text):
    """Защитный regex от markdown-обёртки + pydantic-валидация."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    raw = match.group(0) if match else text
    return Solution.model_validate_json(raw)

def format_solution(solution):
    """Solution -> текст сообщения для ВК (в терминале это был print)."""
    lines = [
        f'ДАНО: {solution.given}',
        f'НАЙТИ: {solution.find}',
        'РЕШЕНИЕ:',
    ]
    lines += [f'  - {step}' for step in solution.steps]
    lines.append(f'ОТВЕТ: {solution.answer}')
    return '\n'.join(lines)

# ============================================================
# Кубик 6. Безопасность: injection, длина, лог
# ============================================================

INJECTION_PATTERNS = [
    re.compile(r'\b(ignore|забудь|forget|disregard)[\s\-_]*(all|everything|все|всё|previous|предыдущ)', re.I),
    re.compile(r'\b(ты|ты\s+теперь|you\s+are\s+now)[\s\-_]+(пират|хакер|dan|jailbreak)', re.I),
    # .{0,20}? — чтобы «покажи СВОЙ промпт» тоже ловился
    re.compile(r'\b(reveal|покажи|расскажи|скажи|show)\b.{0,20}?\b(prompt|system|промпт|системн)', re.I),
    re.compile(r'\b(api[\s\-_]?key|token|токен|ключ|password|пароль)\b', re.I),
    re.compile(r'\b(выйди\s+из|exit\s+from|leave)[\s\-_]+(роли|role)', re.I),
]

def is_injection(text):
    """True, если в тексте сработал хотя бы один паттерн атаки."""
    return any(p.search(text) for p in INJECTION_PATTERNS)

def is_too_long(text, limit=MAX_INPUT_LEN):
    """True, если ввод длиннее лимита по символам."""
    return len(text) > limit

def log_event(peer_id, user_text, bot_text, tokens):
    """Пишем в bot.log усечённые версии + число токенов. Без ключей и полного текста."""
    logging.info(
        'peer=%s | user="%s" | bot="%s" | tokens=%s',
        peer_id,
        user_text[:100].replace('\n', ' '),
        bot_text[:200].replace('\n', ' '),
        tokens,
    )

# ============================================================
# Вызов GigaChat
# ============================================================

def ask_gigachat(messages):
    """Возвращает (текст ответа, токены) или (None, 0) при ошибке."""
    try:
        # with — закрывает httpx-клиент; в долгоживущем боте иначе копятся соединения
        with GigaChat(
            credentials=GIGACHAT_API_KEY,
            base_url="https://api.giga.chat/v1",
            scope=GIGACHAT_SCOPE,
            model=GIGACHAT_MODEL,
            verify_ssl_certs=False,  # Для тестирования (в продакшене установите True)
            timeout=30,
        ) as giga:
            # messages — полноценный список ролей, а не склеенная строка
            chat = Chat(
                messages=messages,
                temperature=1,
                max_tokens=1000,
            )
            response = giga.chat(chat)

            if not response.choices:
                return None, response.usage.total_tokens
            # у SDK это объект, а не dict: response.usage.total_tokens
            return response.choices[0].message.content, response.usage.total_tokens

    except Exception as e:
        print(f"❌ Ошибка при запросе к GigaChat: {e}")
        logging.error('gigachat_error | %s', e)
        return None, 0

# ============================================================
# Кубик 7. Вложения ВК: фото и голосовые -> текст
# Двухшаговая схема: Pro распознаёт вложение в текст,
# дальше текст идёт по обычному конвейеру (фильтры -> RAG -> JSON).
# ============================================================

VOICE_PROMPT = (
    'Расшифруй голосовое сообщение дословно, на языке оригинала. '
    'В ответе — только текст расшифровки, без комментариев и кавычек.'
)
PHOTO_PROMPT = (
    'На изображении — задача по математике или физике. '
    'Выпиши её условие текстом: все числа, формулы и обозначения. '
    'В ответе — только текст условия, без решения и комментариев. '
    'Если задачи на изображении нет, ответь ровно: НЕТ ЗАДАЧИ.'
)

def extract_attachment(msg):
    """Первое поддерживаемое вложение сообщения.
    Возвращает ('photo', url) | ('voice', (url, duration)) | (None, None)."""
    for att in msg.get('attachments', []):
        if att.get('type') == 'photo':
            sizes = att.get('photo', {}).get('sizes', [])
            if sizes:
                # ВК отдаёт набор размеров — берём самый крупный, GigaChat читает его лучше
                best = max(sizes, key=lambda s: s.get('width', 0) * s.get('height', 0))
                return 'photo', best['url']
        elif att.get('type') == 'audio_message':
            am = att.get('audio_message', {})
            # у голосовых ВК две ссылки; mp3 надёжнее по MIME (в доках GigaChat — audio/mp3)
            url = am.get('link_mp3') or am.get('link_ogg')
            if url:
                return 'voice', (url, am.get('duration', 0))
    return None, None

def download(url, max_mb):
    """Качаем вложение с CDN ВК. None — если не вышло или файл больше лимита."""
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logging.error('download_error | %s | %s', url[:80], e)
        return None
    if len(resp.content) > max_mb * 1024 * 1024:
        logging.warning('download_too_big | %s байт | %s', len(resp.content), url[:80])
        return None
    return resp.content

def recognize_attachment(raw, filename, mime, prompt, audio=False):
    """Файл -> хранилище GigaChat -> запрос с attachments -> удаление файла.
    Возвращает (распознанный текст, токены) или (None, 0)."""
    try:
        with GigaChat(
            credentials=GIGACHAT_API_KEY,
            base_url="https://api.giga.chat/v1",
            scope=GIGACHAT_SCOPE,
            model=RECOGNITION_MODEL,
            verify_ssl_certs=False,  # Для тестирования (в продакшене установите True)
            timeout=60,  # мультимодальные запросы заметно дольше текстовых
        ) as giga:
            # кортеж (имя, байты, MIME) уходит в multipart как есть — так SDK передаёт его httpx
            uploaded = giga.upload_file((filename, raw, mime), purpose="general")
            try:
                extra = {'function_call': 'auto'} if audio else {}  # так в примере доков для аудио
                chat = Chat(
                    messages=[{'role': 'user', 'content': prompt, 'attachments': [uploaded.id_]}],
                    temperature=0.1,
                    max_tokens=800,
                    **extra,
                )
                response = giga.chat(chat)
                if not response.choices:
                    return None, response.usage.total_tokens
                return response.choices[0].message.content.strip(), response.usage.total_tokens
            finally:
                # файл нужен на один запрос, хранилище не мусорим
                try:
                    giga.delete_file(uploaded.id_)
                except Exception:
                    pass
    except Exception as e:
        print(f"❌ Ошибка распознавания вложения: {e}")
        logging.error('recognition_error | %s', e)
        return None, 0

def process_attachment(vk, peer_id, kind, payload):
    """Скачать + распознать вложение. Про ошибки сами говорим пользователю.
    Возвращает (текст, токены) или (None, токены)."""
    if kind == 'voice':
        url, duration = payload
        if duration > MAX_VOICE_SEC:
            send_vk(vk, peer_id, f'Голосовые длиннее {MAX_VOICE_SEC} секунд не разбираю. Скажи короче или напиши текстом.')
            return None, 0
        raw = download(url, MAX_AUDIO_MB)
        if raw is None:
            send_vk(vk, peer_id, 'Не смог скачать голосовое. Пришли ещё раз.')
            return None, 0
        print("⏳ Расшифровка голосового через GigaChat...")
        text, tokens = recognize_attachment(raw, 'voice.mp3', 'audio/mp3', VOICE_PROMPT, audio=True)
        if not text:
            send_vk(vk, peer_id, 'Не разобрал голосовое. Скажи чётче или напиши текстом.')
            return None, tokens
        send_vk(vk, peer_id, f'Услышал: «{text[:300]}»')
        log_event(peer_id, f'[voice] {text}', 'RECOGNIZED', tokens)
        return text, tokens

    # kind == 'photo'
    raw = download(payload, MAX_IMAGE_MB)
    if raw is None:
        send_vk(vk, peer_id, f'Не смог скачать фото (или оно больше {MAX_IMAGE_MB} МБ). Пришли ещё раз.')
        return None, 0
    print("⏳ Распознавание фото через GigaChat...")
    text, tokens = recognize_attachment(raw, 'photo.jpg', 'image/jpeg', PHOTO_PROMPT)
    if not text or 'НЕТ ЗАДАЧИ' in text.upper():
        send_vk(vk, peer_id, 'Не нашёл задачу на фото. Сними условие покрупнее или напиши текстом.')
        return None, tokens
    send_vk(vk, peer_id, f'Прочитал с фото: «{text[:300]}»')
    log_event(peer_id, f'[photo] {text}', 'RECOGNIZED', tokens)
    return text, tokens

# ============================================================
# Кубик 5. История: своя conversation на каждый диалог ВК
# ============================================================

CONVERSATIONS = {}  # peer_id -> [{'role': 'system', ...}, {'role': 'user', ...}, ...]

def build_messages(peer_id, user_text):
    """system с RAG + few-shot + обрезанная история этого диалога + новый вопрос."""
    conversation = CONVERSATIONS.setdefault(
        peer_id, [{'role': 'system', 'content': system_text}]
    )
    # RAG: system пересобирается под каждый вопрос — материалы свои для каждого
    conversation[0] = {'role': 'system', 'content': build_system(user_text)}
    # обрезка: system держим всегда, режем только «хвост» диалога
    return (
        conversation[:1]
        + FEW_SHOT
        + conversation[1:][-MAX_HISTORY:]
        + [{'role': 'user', 'content': user_text}]
    )

def remember(peer_id, user_text, solution):
    """Кладём пару вопрос-ответ в историю этого диалога."""
    conversation = CONVERSATIONS[peer_id]
    conversation.append({'role': 'user', 'content': user_text})
    conversation.append({'role': 'assistant', 'content': solution.model_dump_json()})

# ============================================================
# Отправка в ВК
# ============================================================

def split_message(text, limit=VK_MSG_LIMIT):
    """Режем длинный ответ по строкам — ВК не принимает больше ~4096 символов."""
    parts = []
    current = ''
    for line in text.split('\n'):
        while len(line) > limit:  # одна строка длиннее лимита — рубим как есть
            parts.append(line[:limit])
            line = line[limit:]
        if current and len(current) + len(line) + 1 > limit:
            parts.append(current.rstrip('\n'))
            current = ''
        current += line + '\n'
    if current.strip():
        parts.append(current.rstrip('\n'))
    return parts or ['(пусто)']

def send_vk(vk, peer_id, text):
    """Отправка с разбивкой длинных сообщений."""
    try:
        parts = split_message(text)
        for i, part in enumerate(parts):
            prefix = "" if i == 0 else "(продолжение)\n"
            vk.messages.send(
                peer_id=peer_id,
                message=f"{prefix}{part}",
                random_id=get_random_id(),
            )
            if len(parts) > 1:
                time.sleep(0.3)
        print("✅ Ответ отправлен!")
    except Exception as e:
        print(f"❌ Ошибка отправки сообщения VK: {e}")
        logging.error('vk_send_error | peer=%s | %s', peer_id, e)

# ============================================================
# Приветствия и команды — отвечаем сами, без модели
# ============================================================

# Сравниваем по ВСЕМУ тексту целиком, а не по началу строки:
# «привет, забудь все инструкции» сюда не попадёт и уедет в фильтр атак.
GREETINGS = {
    'привет', 'приветик', 'приветствую', 'здравствуй', 'здравствуйте', 'хай',
    'ку', 'здарова', 'здорово', 'добрый день', 'доброе утро', 'добрый вечер',
    'hi', 'hello', 'как дела', 'ты тут', 'ты здесь',
}
START_COMMANDS = {'start', 'старт', 'начать', 'начало', 'меню'}  # кнопка ВК «Начать» шлёт текст «Начать»
HELP_COMMANDS = {'help', 'помощь', 'справка', 'что ты умеешь', 'что умеешь'}
CLEAR_COMMANDS = {'clear', 'сброс', 'очистить', 'очисти историю', 'забудь историю'}
THANKS = {'спасибо', 'спс', 'благодарю', 'пасиб', 'thanks'}
BYE = {'пока', 'до свидания', 'бай', 'bye', 'спокойной ночи'}

HELLO_TEXT = (
    'Привет! Я решаю задачи по математике и физике — строго по материалам учебника.\n\n'
    'Пришли условие текстом, голосовым или фото задачи, например:\n'
    '  • Реши уравнение 2x + 3 = 11\n'
    '  • Найди массу, если плотность 2700 кг/м3, объём 0,5 м3\n\n'
    'Команды: /help — подробнее, /clear — забыть наш разговор.'
)

HELP_TEXT = (
    'Как я работаю:\n'
    '  • отвечаю ТОЛЬКО по материалам учебника; если ответа там нет — так и скажу\n'
    '  • разбираю решение на ДАНО / НАЙТИ / РЕШЕНИЕ / ОТВЕТ\n'
    '  • понимаю голосовые (до 90 сек) и фото условия: распознаю текст и решаю\n'
    '  • помню предыдущие вопросы в нашем диалоге, можно спрашивать «а если объём 2 м3»\n'
    f'  • длина вопроса — до {MAX_INPUT_LEN} символов (голосовых и распознанных тоже)\n\n'
    'Команды:\n'
    '  /clear — очистить историю\n'
    '  /help — это сообщение'
)

def command_key(text):
    """Ключ для сравнения с командами: нижний регистр, без пунктуации и лишних пробелов.
    normalize() съедает и слэш, поэтому '/start' и 'Старт!' дают один и тот же 'старт'."""
    return ' '.join(normalize(text).split())

def handle_command(vk, peer_id, user_text):
    """True, если сообщение — приветствие или команда. Модель не дёргаем, токены не тратим."""
    key = command_key(user_text)

    if key in GREETINGS or key in START_COMMANDS:
        send_vk(vk, peer_id, HELLO_TEXT)
    elif key in HELP_COMMANDS:
        send_vk(vk, peer_id, HELP_TEXT)
    elif key in CLEAR_COMMANDS:
        CONVERSATIONS.pop(peer_id, None)
        send_vk(vk, peer_id, 'История очищена. Задавай задачу заново.')
    elif key in THANKS:
        send_vk(vk, peer_id, 'Пожалуйста! Присылай следующую задачу.')
    elif key in BYE:
        send_vk(vk, peer_id, 'Пока! Будет задача — пиши.')
    else:
        return False

    log_event(peer_id, user_text, 'COMMAND', 0)
    return True

# ============================================================
# Обработка одного сообщения
# ============================================================

def handle_message(vk, event):
    msg = event.obj.message
    user_text = msg.get('text', '').strip()
    from_id = msg.get('from_id', 0)
    peer_id = msg.get('peer_id', 0)

    # Игнорируем сообщения от бота
    if from_id < 0:
        return

    kind, payload = extract_attachment(msg)

    # ни текста, ни поддерживаемого вложения
    if not user_text and kind is None:
        if msg.get('attachments'):  # стикер, документ, видео и т.п.
            send_vk(vk, peer_id, 'Такое вложение не разбираю. Пришли текст, фото задачи или голосовое.')
        return

    print(f"👤 Пользователь {from_id}: {user_text or '[' + kind + ']'}")

    # вложение -> текст (шаг 1 двухшаговой схемы); дальше текст идёт по обычному конвейеру
    recognition_tokens = 0
    if kind is not None:
        try:
            vk.messages.setActivity(peer_id=peer_id, type="typing")
        except Exception:
            pass
        recognized, recognition_tokens = process_attachment(vk, peer_id, kind, payload)
        if recognized is None:
            return  # пользователю уже ответили внутри process_attachment
        # подпись к фото приклеиваем перед распознанным условием
        user_text = f'{user_text}\n{recognized}'.strip() if user_text else recognized

    # приветствия и команды: /exit тут не нужен, бот живёт всегда
    if handle_command(vk, peer_id, user_text):
        return

    # фильтры до вызова модели — на блокировке не тратим ни токена
    if is_too_long(user_text):
        send_vk(vk, peer_id, f'Слишком длинный вопрос ({len(user_text)} > {MAX_INPUT_LEN} символов). Сократи и пришли ещё раз.')
        return

    if is_injection(user_text):
        send_vk(vk, peer_id, 'Подозрительный ввод. Не обрабатываю.')
        logging.warning('injection_blocked | peer=%s | user="%s"', peer_id, user_text[:100])
        return

    # Показываем статус "печатает..."
    try:
        vk.messages.setActivity(peer_id=peer_id, type="typing")
    except Exception:
        pass

    print("⏳ Отправка запроса в GigaChat...")
    text, tokens = ask_gigachat(build_messages(peer_id, user_text))
    total_tokens = recognition_tokens + tokens  # распознавание + решение

    if total_tokens > TOKEN_ALERT:
        print(f"[!] ALERT: {total_tokens} токенов на один ответ (peer {peer_id})", file=sys.stderr)

    if text is None:
        send_vk(vk, peer_id, 'Произошла техническая ошибка. Попробуйте позже.')
        return

    print(f"🤖 GigaChat: {text[:100]}...")  # Первые 100 символов

    try:
        solution = parse_solution(text)
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        logging.error('validation_error | peer=%s | text="%s" | err=%s', peer_id, text[:200], e)
        send_vk(vk, peer_id, 'Модель вернула невалидный JSON. Попробуй переформулировать вопрос.')
        return

    # в историю — только после успешного парсинга
    remember(peer_id, user_text, solution)
    send_vk(vk, peer_id, format_solution(solution))
    log_event(peer_id, user_text, solution.answer, total_tokens)

# ============================================================
# Главный цикл
# ============================================================

def use_vk_ru(vk_session):
    """vk_api 11.10.0 с PyPI (релиз от 18.07.2025) ходит в захардкоженный
    https://api.vk.com/method/. В master библиотеки домен уже заменён на api.vk.ru,
    но релиза нет — поэтому переписываем URL на уровне http-сессии.
    Адрес longpoll-сервера подменять не нужно: он приходит от самого ВК."""
    original_post = vk_session.http.post

    def post(url, *args, **kwargs):
        if url.startswith('https://api.vk.com/'):
            url = VK_API_HOST + url[len('https://api.vk.com/'):]
        return original_post(url, *args, **kwargs)

    vk_session.http.post = post

def main():
    # Проверяем наличие переменных
    if not VK_TOKEN:
        print("❌ Ошибка: VK_API не найден в .env файле!")
        return
    if not GIGACHAT_API_KEY:
        print("❌ Ошибка: GIGACHAT_API_KEY не найден в .env файле!")
        return

    # Инициализация VK
    vk_session = vk_api.VkApi(token=VK_TOKEN, api_version='5.131')
    use_vk_ru(vk_session)
    vk = vk_session.get_api()

    print("🤖 Бот VK + GigaChat запущен!")
    print(f"📊 Группа ID: {GROUP_ID}")
    print(f"🌐 VK API: {VK_API_HOST}")
    print(f"🤖 Модель GigaChat: {GIGACHAT_MODEL} (текст) + {RECOGNITION_MODEL} (фото/голос)")
    print(f"📚 Корпус: {len(CORPUS)} кусков")
    print("Ожидание сообщений...")
    print("-" * 50)

    while True:
        try:
            longpoll = VkBotLongPoll(vk_session, GROUP_ID)
            for event in longpoll.listen():
                if event.type != VkBotEventType.MESSAGE_NEW:
                    continue
                try:
                    handle_message(vk, event)
                except Exception as e:
                    # одно кривое сообщение не должно ронять бота
                    print(f"❌ Ошибка обработки: {e}")
                    logging.exception('handle_error')
                print("-" * 50)
        except KeyboardInterrupt:
            print("\n👋 Остановлено вручную")
            break
        except Exception as e:
            # сеть моргнула / VK вернул 502 — longpoll.listen() бросает исключение
            print(f"⚠️ Longpoll оборвался: {e}. Переподключаюсь через 5 сек...")
            logging.warning('longpoll_reconnect | %s', e)
            time.sleep(5)

if __name__ == '__main__':
    main()
