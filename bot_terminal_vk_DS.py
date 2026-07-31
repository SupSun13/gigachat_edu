import os
import re
import sys
import json
import time
import logging
from pathlib import Path

import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.utils import get_random_id
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем переменные из .env файла
load_dotenv()

# --- КОНФИГУРАЦИЯ ---
VK_TOKEN = os.getenv('VK_API')
GROUP_ID = int(os.getenv('GROUP_ID'))
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# ВК требует переводить API-интеграции на .ru (дедлайн был 30.09.2025).
# Верни 'https://api.vk.com/', если .ru вдруг заупрямится.
VK_API_HOST = 'https://api.vk.ru/'

# Настройки DeepSeek
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"

MAX_INPUT_LEN = 500     # символов на пользовательский ввод
MAX_HISTORY = 10        # последних сообщений диалога (помимо system)
TOKEN_ALERT = 1000      # алерт в stderr при превышении токенов на ответ
VK_MSG_LIMIT = 4000     # у ВК потолок ~4096 символов на сообщение

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
# Вызов DeepSeek
# ============================================================

def ask_deepseek(messages):
    """Возвращает (текст ответа, токены) или (None, 0) при ошибке."""
    if not DEEPSEEK_API_KEY:
        print("❌ API ключ не найден. Проверьте .env файл (DEEPSEEK_API_KEY)")
        logging.error('deepseek_error | no api key')
        return None, 0

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }

    # messages — полноценный список ролей, а не склеенная строка
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 1,
        "max_tokens": 1000,
        "stream": False,
        # нативный JSON-режим: модель не сможет вернуть текст вокруг объекта
        "response_format": {"type": "json_object"},
        # у V4 режим размышления опциональный. Для школьных задач выключен.
        # Для сложных: {"type": "enabled"} + "reasoning_effort": "high"
        "thinking": {"type": "disabled"},
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # число токенов из JSON-тела; при отсутствии — 0
        tokens = data.get("usage", {}).get("total_tokens", 0)

        if not data.get("choices"):
            return None, tokens
        return data["choices"][0]["message"]["content"], tokens

    except Exception as e:
        print(f"❌ Ошибка при запросе к DeepSeek: {e}")
        logging.error('deepseek_error | %s', e)
        return None, 0

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
    'Просто пришли условие, например:\n'
    '  • Реши уравнение 2x + 3 = 11\n'
    '  • Найди массу, если плотность 2700 кг/м3, объём 0,5 м3\n\n'
    'Команды: /help — подробнее, /clear — забыть наш разговор.'
)

HELP_TEXT = (
    'Как я работаю:\n'
    '  • отвечаю ТОЛЬКО по материалам учебника; если ответа там нет — так и скажу\n'
    '  • разбираю решение на ДАНО / НАЙТИ / РЕШЕНИЕ / ОТВЕТ\n'
    '  • помню предыдущие вопросы в нашем диалоге, можно спрашивать «а если объём 2 м3»\n'
    f'  • длина вопроса — до {MAX_INPUT_LEN} символов\n\n'
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

    # Игнорируем сообщения от бота и пустые сообщения
    if from_id < 0 or not user_text:
        return

    print(f"👤 Пользователь {from_id}: {user_text}")

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

    print("⏳ Отправка запроса в DeepSeek...")
    text, tokens = ask_deepseek(build_messages(peer_id, user_text))

    if tokens > TOKEN_ALERT:
        print(f"[!] ALERT: {tokens} токенов на один ответ (peer {peer_id})", file=sys.stderr)

    if text is None:
        send_vk(vk, peer_id, 'Произошла техническая ошибка. Попробуйте позже.')
        return

    print(f"🤖 DeepSeek: {text[:100]}...")  # Первые 100 символов

    try:
        solution = parse_solution(text)
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        logging.error('validation_error | peer=%s | text="%s" | err=%s', peer_id, text[:200], e)
        send_vk(vk, peer_id, 'Модель вернула невалидный JSON. Попробуй переформулировать вопрос.')
        return

    # в историю — только после успешного парсинга
    remember(peer_id, user_text, solution)
    send_vk(vk, peer_id, format_solution(solution))
    log_event(peer_id, user_text, solution.answer, tokens)

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
    if not DEEPSEEK_API_KEY:
        print("❌ Ошибка: DEEPSEEK_API_KEY не найден в .env файле!")
        return

    # Инициализация VK
    vk_session = vk_api.VkApi(token=VK_TOKEN, api_version='5.131')
    use_vk_ru(vk_session)
    vk = vk_session.get_api()

    print("🤖 Бот VK + DeepSeek запущен!")
    print(f"📊 Группа ID: {GROUP_ID}")
    print(f"🌐 VK API: {VK_API_HOST}")
    print(f"🤖 Модель DeepSeek: {DEEPSEEK_MODEL}")
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
