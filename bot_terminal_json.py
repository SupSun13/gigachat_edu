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

def ask_gigachat(question):
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
            base_url="https://api.giga.chat/v1",  # Явно указываем URL
            verify_ssl_certs=False,
            timeout=30,
        )
        

        # Отправляем запрос
        history = [{"role": "system", "content": system_text}]
        history += FEW_SHOT
        history += [{"role": "user", "content": question}]
        chat = Chat(
            model="GigaChat-2",
            messages = history,
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
    print("Введите 'exit' для выхода\n")
    
    while True:
        question = input("👤 Вы: ").strip()
        
        if question.lower() in ["exit", "quit"]:
            print("👋 До свидания!")
            break
        
        if not question:
            print("❌ Пожалуйста, введите вопрос")
            continue
        
        print("🤖 GigaChat: ", end="")
        text = ask_gigachat(question)
        print(text)
        try:
            solution = parse_solution(text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            print('Модель вернула невалидный JSON. Попробуй переформулировать.')
            continue
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
