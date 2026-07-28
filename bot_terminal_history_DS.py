import os
import json
import re
from typing import List

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

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

MAX_HISTORY = 10


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


def ask_deepseek(messages):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "❌ API ключ не найден. Проверьте .env файл (DEEPSEEK_API_KEY)"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 1,
        "max_tokens": 1000,
        "stream": False,
        # нативный JSON-режим: модель физически не сможет вернуть текст вокруг объекта
        "response_format": {"type": "json_object"},
        # V4 умеет "думать"; для школьных задач выключаем — быстрее и дешевле.
        # Для сложных задач: {"type": "enabled"} + "reasoning_effort": "high"
        "thinking": {"type": "disabled"},
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices")
        if not choices:
            return "Нет ответа от модели"

        usage = data.get("usage", {})
        print("токенов потрачено:", usage.get("total_tokens", "?"))
        return choices[0]["message"]["content"]

    except requests.HTTPError as e:
        return f"Ошибка HTTP {e.response.status_code}: {e.response.text}"
    except requests.RequestException as e:
        return f"Ошибка сети: {e}"
    except (KeyError, ValueError) as e:
        return f"Ошибка разбора ответа: {e}"


def main():
    print("🤖 DeepSeek терминал")
    print("=" * 50)
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
        #    few-shot и текущий вопрос кладём поверх
        messages = (
            conversation[:1]
            + FEW_SHOT
            + conversation[1:][-MAX_HISTORY:]
            + [{'role': 'user', 'content': question}]
        )
        print("🤖 DeepSeek: ", end="")
        # 3. вызываем модель
        text = ask_deepseek(messages)
        print(text)

        try:
            solution = parse_solution(text)
        except (ValidationError, ValueError, json.JSONDecodeError):
            print('Модель вернула невалидный JSON. Попробуй переформулировать.')
            continue

        # 3. сохраняем вопрос и ответ в историю (только после успешного парсинга)
        conversation.append({'role': 'user', 'content': question})
        conversation.append({'role': 'assistant', 'content': solution.model_dump_json()})
        print("len" + str(len(messages)))

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
