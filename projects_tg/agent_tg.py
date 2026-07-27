import asyncio
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv(Path(__file__).parent / ".env")   # ключи — ДО импорта агента

from gigachat_agent.agent import root_agent

APP_NAME = "weather_bot"

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)


async def ensure_session(chat_id: int) -> str:
    sid = f"tg-{chat_id}"
    s = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=sid, session_id=sid
    )
    if s is None:
        await runner.session_service.create_session(
            app_name=APP_NAME, user_id=sid, session_id=sid
        )
    return sid


async def ask_agent(chat_id: int, text: str) -> str:
    sid = await ensure_session(chat_id)
    msg = types.Content(role="user", parts=[types.Part(text=text)])

    answer = ""
    async for event in runner.run_async(
        user_id=sid, session_id=sid, new_message=msg
    ):
        if event.get_function_calls():
            await bot.send_message(chat_id, "🌤 Смотрю погоду…")
        if event.is_final_response() and event.content and event.content.parts:
            answer = event.content.parts[0].text or ""
    return answer or "Модель не вернула ответ."


@dp.message(Command("start"))
async def cmd_start(message: Message):
    await ensure_session(message.chat.id)
    await message.answer("Привет! Спроси меня про погоду в любом городе.")


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    sid = f"tg-{message.chat.id}"
    try:
        await runner.session_service.delete_session(
            app_name=APP_NAME, user_id=sid, session_id=sid
        )
    except Exception:
        pass
    await message.answer("Память диалога очищена.")


@dp.message(F.text)
async def on_message(message: Message):
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        answer = await ask_agent(message.chat.id, message.text)
    except Exception as e:
        await message.answer(f"Ошибка агента: {e}")
        return
    await message.answer(answer)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
