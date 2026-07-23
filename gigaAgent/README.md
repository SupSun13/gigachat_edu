# GigaChat Agent на Google ADK

Учебный проект: ИИ-агент на [Google Agent Development Kit](https://google.github.io/adk-docs/) с моделью **GigaChat** вместо Gemini.

Агент умеет вызывать инструменты (function calling). В примере — функция погоды, которая дёргает публичный API [Open-Meteo](https://open-meteo.com/): модель сама определяет координаты города, вызывает функцию и отвечает по её данным.

Как это соединяется:

```
ADK  →  LiteLLM (переводчик)  →  GigaChat API
```

ADK нативно работает только с Gemini, поэтому внешние модели подключаются через обёртку `LiteLlm`. В LiteLLM есть нативный провайдер `gigachat/` — отдельный GigaChat SDK не нужен, достаточно ключа.

## Требования

- Python 3.10+
- Ключ GigaChat API (физлицам — бесплатный пакет токенов)

## Установка

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO

python -m venv venv
source venv/bin/activate        # Linux / macOS
# source venv/Scripts/activate  # Windows (Git Bash)

pip install google-adk litellm requests python-dotenv
pip install -U litellm          # поддержка GigaChat добавлена недавно, нужна свежая версия
```

## Ключ GigaChat

1. Зарегистрируйся на [developers.sber.ru](https://developers.sber.ru/) (вход через Сбер ID).
2. Создай проект **GigaChat API**.
3. В настройках проекта получи **Authorization Key** — длинную base64-строку. Скоуп для физлица — `GIGACHAT_API_PERS`.

Создай файл `gigachat_agent/.env` и впиши ключ:

```
GIGACHAT_CREDENTIALS=твоя_base64_строка_целиком
```

> **Не коммить `.env`!** Файл уже в `.gitignore`. Ключ, попавший в публичный репозиторий, считается скомпрометированным — его нужно отозвать и создать новый.

## Запуск

Из **корня репозитория** (не из папки агента):

```bash
adk web
```

Открой [http://localhost:8000](http://localhost:8000), выбери `gigachat_agent` и спроси, например: *«Какая погода в Казани?»* В интерфейсе видно каждый шаг агентного цикла: вызов функции, её ответ, финальный текст модели.

Вариант без браузера:

```bash
adk run gigachat_agent
```

## Структура проекта

```
.
├── gigachat_agent/
│   ├── __init__.py     # from . import agent
│   ├── agent.py        # определение агента: модель + инструменты
│   └── .env            # ключ (не в репозитории)
├── .gitignore
└── README.md
```

## Частые ошибки

**`SSL: CERTIFICATE_VERIFY_FAILED`** — GigaChat API работает на сертификатах НУЦ Минцифры, которых нет в стандартных хранилищах. В коде стоит `ssl_verify=False` — это допустимо для учебного демо. Правильный путь для продакшена — установить [сертификаты Минцифры](https://www.gosuslugi.ru/crt) и убрать флаг.

**`401 Unauthorized`** — проверь, что в `.env` вставлен именно **Authorization Key** целиком (base64-строка), а не client_id или client_secret по отдельности.

**Ключ `None` / агент не видит ключ** — файл `.env` лежит не в папке агента, или запуск выполнен не из корня репозитория.

**`adk: command not found`** — не активировано виртуальное окружение (`source venv/...`).

## Ограничения

- Встроенные инструменты ADK (например, `google_search`) работают только с Gemini. С GigaChat используются свои функции-инструменты — как `get_weather` в этом проекте.
- Список доступных моделей (`GigaChat-2-Lite`, `GigaChat-2-Max`, …) зависит от тарифа твоего API-доступа.

## Полезные ссылки

- [Документация ADK: подключение моделей через LiteLLM](https://google.github.io/adk-docs/agents/models/litellm/)
- [LiteLLM: провайдер GigaChat](https://docs.litellm.ai/docs/providers/gigachat)
- [Документация GigaChat API](https://developers.sber.ru/docs/ru/gigachat/api/overview)
- [gpt2giga](https://github.com/ai-forever/gpt2giga) — прокси Сбера, если нужен OpenAI/Anthropic-совместимый доступ к GigaChat
