import os
import sys
import logging
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat
import json
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

MAX_INPUT_LEN = 500     # символов на пользовательский ввод
MAX_HISTORY = 10        # последних сообщений диалога (помимо system)
TOKEN_ALERT = 1000      # алерт в stderr при превышении токенов на ответ

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

# ============================================================
# Кубик 6. Безопасность: injection, длина, лог
# ============================================================

INJECTION_PATTERNS = [
    re.compile(r'\b(ignore|забудь|forget|disregard)[\s\-_]*(all|everything|все|всё|previous|предыдущ)', re.I),
    re.compile(r'\b(ты|ты\s+теперь|you\s+are\s+now)[\s\-_]+(пират|хакер|dan|jailbreak)', re.I),
    # .{0,20}? — чтобы «покажи СВОЙ промпт» тоже ловился (см. комментарий ниже)
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

def log_event(user_text, bot_text, tokens):
    """Пишем в bot.log усечённые версии + число токенов. Без личных данных и ключей."""
    logging.info(
        'user="%s" | bot="%s" | tokens=%s',
        user_text[:100].replace('\n', ' '),
        bot_text[:200].replace('\n', ' '),
        tokens,
    )

# ============================================================
# Вызов модели
# ============================================================

def ask_gigachat(messages):
    """Возвращает (текст ответа, число токенов)."""
    # Получаем ключ
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        credentials = os.getenv("GIGACHAT_API_KEY")

    if not credentials:
        return "❌ API ключ не найден. Проверьте .env файл", 0

    try:
        # Создаем клиент с явным указанием URL
        client = GigaChat(
            credentials=credentials,
            base_url="https://gigachat.devices.sberbank.ru/api/v1",  # Явно указываем URL
            verify_ssl_certs=False,
            timeout=30,
        )

        # messages уже собраны в main(): system+RAG + few-shot + история + новый вопрос
        chat = Chat(
            messages=messages,
            temperature=1,
            max_tokens=1000,
        )

        response = client.chat(chat)

        if response.choices:
            # у SDK это объект, а не dict: response.usage.total_tokens
            return response.choices[0].message.content, response.usage.total_tokens
        else:
            return "Нет ответа от модели", response.usage.total_tokens

    except Exception as e:
        return f"Ошибка: {str(e)}", 0

# ============================================================
# Кубик 5. Главный цикл: conversation с обрезкой
# ============================================================

def main():
    print("🤖 GigaChat терминал")
    print("="*50)
    print("Введите '/exit' для выхода\n")

    # conversation инициализируется системным сообщением
    conversation = [{'role': 'system', 'content': system_text}]

    while True:
        question = input("👤 Вы: ").strip()

        if question.lower() in ["/exit", "exit", "quit"]:
            print("👋 До свидания!")
            break

        if not question:
            print("❌ Пожалуйста, введите вопрос")
            continue

        # фильтры до вызова модели — на блокировке не тратим ни токена
        if is_too_long(question):
            print(f'Слишком длинный ввод ({len(question)} > {MAX_INPUT_LEN}). Сократи и попробуй ещё раз.')
            continue

        if is_injection(question):
            print('Подозрительный ввод. Не обрабатываю.')
            logging.warning('injection_blocked | user="%s"', question[:100])
            continue

        # RAG: system пересобирается под каждый вопрос — материалы свои для каждого
        conversation[0] = {'role': 'system', 'content': build_system(question)}

        # обрезка: system держим всегда, режем только «хвост» диалога
        messages = (
            conversation[:1]
            + FEW_SHOT
            + conversation[1:][-MAX_HISTORY:]
            + [{'role': 'user', 'content': question}]
        )

        print("🤖 GigaChat: ", end="")
        text, tokens = ask_gigachat(messages)
        print(text)

        if tokens > TOKEN_ALERT:
            print(f'[!] Использовано {tokens} токенов на один ответ.', file=sys.stderr)

        try:
            solution = parse_solution(text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            print('Модель вернула невалидный JSON. Попробуй переформулировать.')
            logging.error('validation_error | text="%s" | err=%s', text[:200], e)
            continue

        # сохраняем вопрос и ответ в историю (только после успешного парсинга)
        conversation.append({'role': 'user', 'content': question})
        conversation.append({'role': 'assistant', 'content': solution.model_dump_json()})

        # вывод
        print(f'\nДАНО: {solution.given}')
        print(f'НАЙТИ: {solution.find}')
        print('РЕШЕНИЕ:')
        for step in solution.steps:
            print(f'  - {step}')
        print(f'ОТВЕТ: {solution.answer}\n')

        # лог
        log_event(question, solution.answer, tokens)

if __name__ == "__main__":
    main()
