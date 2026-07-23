import os, requests
from dotenv import load_dotenv
load_dotenv()
api_key= os.getenv('DEEPSEEK_API_KEY')

# question = input('Введи запрос: ')
response = requests.post(
    'https://api.deepseek.com/chat/completions',
    headers={'Authorization': f'Bearer {api_key}'},
    json={'model': 'deepseek-v4-flash',
    'messages': [
        {"role": "system",
        "content": "Отвечай в формате "
                    "ДАНО / НАЙТИ / РЕШЕНИЕ / ОТВЕТ. "
                    "Никаких лишних слов до и после."},
        {"role":"user","content":"реши x + 2 = 5"},
        {"role":"assistant",
        "content":"ДАНО: x+2=5\nНАЙТИ: x\n"
                "РЕШЕНИЕ: x=5-2\nОТВЕТ: x=3"},
        {"role":"user","content":"реши 2y = 10"},
        {"role":"assistant",
        "content":"ДАНО: 2y=10\nНАЙТИ: y\n"
                "РЕШЕНИЕ: y=10/2\nОТВЕТ: y=5"},
        {"role":"user","content":"реши 2 + 7x = 765"}
        ],
    'temperature': 1.0},
    )
print(api_key[:10] + '...')
print(response.json())
print(response.json()['choices'][0]['message']['content'])
