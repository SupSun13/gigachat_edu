import os
import requests
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from datetime import datetime



load_dotenv()  # подстраховка, чтобы ключ точно подхватился

def get_weather(latitude: float, longitude: float) -> dict:
    """Возвращает текущую температуру по координатам (широта, долгота)."""
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m",
        },
        timeout=10,
    )
    return r.json().get("current", {})


def get_current_time() -> str:
    """Возвращает время в формате ЧЧ:ММ:СС"""
    return datetime.now().strftime("%H:%M:%S")



root_agent = Agent(
    name="gigachat_agent",
    model=LiteLlm(
        model="gigachat/GigaChat-2",
        api_key=os.getenv("GIGACHAT_CREDENTIALS"),
        ssl_verify=False,   # см. раздел про сертификаты
    ),
    description="Агент на GigaChat, умеет узнавать погоду.",
    instruction=(
        "Отвечай кратко и по-русски. "
        "Для вопросов о погоде вызывай инструмент get_weather, "
        "координаты города определяй сам."
    ),
    tools=[get_weather, get_current_time],
)