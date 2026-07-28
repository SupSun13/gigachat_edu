import os
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

# ============================================================
# Корпус и TF-IDF индекс (RAG)
# ============================================================

CORPUS_PATH = Path(__file__).resolve().parent / 'corpus.txt'

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

def retrieve(query, k=2):
    """TF-IDF + cosine similarity. Возвращает k самых релевантных параграфов."""
    q_vec = VECTORIZER.transform([query])
    sims = cosine_similarity(q_vec, MATRIX).ravel()
    top_idx = sims.argsort()[-k:][::-1]
    return [CORPUS[i] for i in top_idx]

# ============================================================
# System-промпт с контекстом
# ============================================================

system_text = (
    'Ты учитель математики и физики. '
    'Отвечай СТРОГО в JSON-формате: '
    '{"given": "...", "find": "...", "steps": [...], "answer": "..."}. '
    'Никакого текста до и после, никаких тройных кавычек. '
    'Используй ТОЛЬКО материалы ниже, если они подходят к вопросу. '
    'Если в материалах нет ответа — всё равно реши задачу, опираясь на школьные знания, но укажи, что материалов нет в скобках в ДАНО.\n\n'
)

def build_system(user_text):
    """system_text + top-2 куска корпуса. Конкатенация, а не .format():
    в шаблоне есть фигурные скобки JSON, .format() на них падает."""
    context = '\n\n'.join(retrieve(user_text, k=2))
    return system_text + 'Материалы:\n' + context

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

MAX_HISTORY = 10

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

def ask_gigachat(messages):
    # Получаем ключ
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        credentials = os.getenv("GIGACHAT_API_KEY")

    if not credentials:
        return "❌ API ключ не найден. Проверьте .env файл"

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
            return response.choices[0].message.content
        else:
            return "Нет ответа от модели"

    except Exception as e:
        return f"Ошибка: {str(e)}"

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
        text = ask_gigachat(messages)
        print(text)

        try:
            solution = parse_solution(text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            print('Модель вернула невалидный JSON. Попробуй переформулировать.')
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
        print()

if __name__ == "__main__":
    main()
