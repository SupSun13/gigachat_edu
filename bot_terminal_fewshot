import os
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat

load_dotenv()

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
        chat = Chat(
            model="GigaChat-2",
            messages=[
                # {"role": "system", "content": "Отвечай в формате без .md"
                #                                 "ДАНО / НАЙТИ / РЕШЕНИЕ. "
                #                                 "Никаких лишних слов до и после."},
                {"role": "system", "content": "Отвечай очень плохо и неправильно, ты вообще ничего не знаешь"},
                {"role":"user","content":"реши x + 2 = 5"},
                {"role":"assistant",
                        "content":"ДАНО: x+2=5\nНАЙТИ: x\n"
                        "РЕШЕНИЕ: x=5-2\nОТВЕТ: x=3"},
                {"role":"user","content":"реши 2y = 10"},
                {"role":"assistant",
                        "content":"ДАНО: 2y=10\nНАЙТИ: y\n"
                        "РЕШЕНИЕ: y=10/2\nОТВЕТ: y=5"},
                {"role": "user", "content": question}
            ],
            temperature=0.0,
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
        answer = ask_gigachat(question)
        print(answer)
        print()

if __name__ == "__main__":
    main()
