import asyncio
import os

import litellm
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()

SYSTEM_PROMPT = "Ты дружелюбный помощник. Отвечай кратко и по-русски."

MAX_TURNS = 20                      # сколько последних реплик помним
history: dict[int, list[dict]] = {}  # chat_id -> список сообщений


@dp.message(Command("start"))
async def cmd_start(message: Message):
    history[message.chat.id] = []
    await message.answer("Привет! Я бот на GigaChat. Спрашивай.")


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    history[message.chat.id] = []
    await message.answer("Память диалога очищена.")


@dp.message(F.text)
async def on_message(message: Message):
    chat_id = message.chat.id

    msgs = history.setdefault(chat_id, [])
    msgs.append({"role": "user", "content": message.text})

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        resp = await litellm.acompletion(
            model="gigachat/GigaChat-2-Max",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                     + msgs[-MAX_TURNS:],
            api_key=os.getenv("GIGACHAT_CREDENTIALS"),
            ssl_verify=False,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        await message.answer(f"Ошибка при запросе к модели: {e}")
        return

    msgs.append({"role": "assistant", "content": answer})
    await message.answer(answer)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
