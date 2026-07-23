import os
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat

load_dotenv()

def ask_gigachat(question):
    gigachat_api = os.getenv("GIGACHAT_API_KEY")
    if not gigachat_api:
        return "Error: API is not in programm!"
    
    try:
        client = GigaChat(
            credentials=gigachat_api,
            base_url="https://api.giga.chat/v1",
            verify_ssl_certs=False,
            timeout=30
        )
        
        chat = Chat(
            model='GigaChat-2',
            messages=[
                {"role": "system", "content": "Ты - учитель по математике, отвечаешь шагами последовательно"
                                                "отвечай строго в формате ДАНО / НАЙТИ / РЕШЕНИЕ / ОТВЕТ"
                                                "НИКАКИХ ЛИШНИХ СЛОВ ДО И ПОСЛЕ"},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        response = client.chat(chat)
        
        if response.choices:
            return response.choices[0].message.content
        else:
            return "There's no answer from model"
        
    except Exception as e:
        return f"Error: {str(e)}"
    
def main():
    print("жижачат запущен!!!!!")
    print('=' * 50)
    print('введи exit, чтобы оборвать чат')
    
    while True:
        question = input('твой запрос: ')
        
        if question.lower() in ['exit', '/exit', 'quit']:
            print('Адьос!')
            break
        
        if not question:
            print('задай вопрос-то, а')
            continue
        
        print('Жижачат говорит: ', end='')
        answer = ask_gigachat(question)
        print(answer)
        print()
        
if __name__ == '__main__':
    print('запуск!')
    main()