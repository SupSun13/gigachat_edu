import os
from dotenv import load_dotenv
import requests
import json
from pydantic import BaseModel, Field, ValidationError
import re

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"

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

def ask_deepseek(question):
    # Получаем ключ
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        return "❌ API ключ не найден. Проверьте .env файл (DEEPSEEK_API_KEY)"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Собираем запрос
    history = [{"role": "system", "content": system_text}]
    history += FEW_SHOT
    history += [{"role": "user", "content": question}]

    payload = {
        "model": MODEL,
        "messages": history,
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
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get("choices"):
            return data["choices"][0]["message"]["content"]
        else:
            return "Нет ответа от модели"

    except requests.HTTPError as e:
        return f"Ошибка HTTP {e.response.status_code}: {e.response.text}"
    except requests.RequestException as e:
        return f"Ошибка сети: {e}"
    except (KeyError, ValueError) as e:
        return f"Ошибка разбора ответа: {e}"

def main():
    print("🤖 DeepSeek терминал")
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

        print("🤖 DeepSeek: ", end="")
        text = ask_deepseek(question)
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
