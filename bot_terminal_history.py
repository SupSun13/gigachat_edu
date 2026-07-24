import os
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat
import json
from pydantic import BaseModel, Field, ValidationError
import re

load_dotenv()

system_text = (
    'Ты учитель математики и физики. '
    'Отвечай СТРОГО в JSON-формате: '
    '{"given": "...", "find": "...", "steps": [...], "answer": "..."}. '
    'Никакого текста до и после, никаких тройных кавычек. '
)

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
from typing import List
class Solution(BaseModel):
    given: str = Field(..., description='что дано в задаче')
    find: str = Field(..., description='что найти')
    steps: List[str] = Field(..., min_length=1, description='шаги решения')
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

        # messages уже собраны в main(): system + few-shot + история + новый вопрос
        chat = Chat(
            messages=messages,
            temperature=1,
            max_tokens=1000,
        )

        response = client.chat(chat)
    
        if response.choices:
            tokens = response.usage.total_tokens
            print("токенов потрачено:", tokens)
            return response.choices[0].message.content
        else:
            return "Нет ответа от модели"

    except Exception as e:
        return f"Ошибка: {str(e)}"

def main():
    print("🤖 GigaChat терминал")
    print("="*50)
    print("Введите '/exit' для выхода\n")

    # 1. conversation инициализируется системным сообщением
    conversation = [{'role': 'system', 'content': system_text}]

    while True:
        # 2. читаем ввод в цикле
        question = input("👤 Вы: ").strip()

        # 5. выход по /exit
        if question.lower() in ["/exit", "exit", "quit"]:
            print("👋 До свидания!")
            break
        
        if question.lower() in ["/clear", "/cl", "clear", "cl"]:
            conversation = [conversation[0]]
            print("История сброшена!")
            continue  

        if not question:
            print("❌ Пожалуйста, введите вопрос")
            continue

        # 4. обрезка: system держим всегда, режем только «хвост» диалога,
        #    few-shot и текущий вопрос кладём поверх (как в v1)
        messages = (
            conversation[:1]
            + FEW_SHOT
            + conversation[1:][-MAX_HISTORY:]
            + [{'role': 'user', 'content': question}]
        )
        print("🤖 GigaChat: ", end="")
        # 3. вызываем модель
        text = ask_gigachat(messages)
        print(text)

        try:
            solution = parse_solution(text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            print('Модель вернула невалидный JSON. Попробуй переформулировать.')
            continue
        

        # 3. сохраняем вопрос и ответ в историю (только после успешного парсинга)
        conversation.append({'role': 'user', 'content': question})
        conversation.append({'role': 'assistant', 'content': solution.model_dump_json()})
        print("len"+ str(len(messages)))

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
