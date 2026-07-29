# Деплой VK + GigaChat бота на Timeweb Cloud

Гайд для бота на long-poll: он не слушает порт, а просто живёт фоновым процессом.
Итог — бот работает круглосуточно, поднимается сам после падения и после перезагрузки сервера.

Везде ниже: `ТВОЙ_IP` — публичный IPv4 сервера, `bot_terminal_vk.py` — имя главного файла (подставь своё).

---

## Что должно получиться

```
/home/bot/vkbot/
├── .venv/                    # виртуальное окружение
├── bot_terminal_vk.py        # код бота
├── corpus.txt                # корпус для RAG
├── requirements.txt          # зависимости
├── .env                      # секреты (chmod 600)
└── bot.log                   # лог, ротируется еженедельно

/etc/systemd/system/vkbot.service    # служба
/etc/logrotate.d/vkbot               # ротация лога
```

---

## Шаг 0. Подготовка на локальной машине

Делается **до** создания сервера — сэкономит время потом.

### 0.1. requirements.txt

`pip freeze > requirements.txt` из conda-окружения (`(base)` в приглашении) не подойдёт: туда попадут сотни лишних пакетов. Собери руками.

Узнать свои версии:

```bash
pip freeze | grep -iE '^(vk[-_]api|python-dotenv|gigachat|pydantic|scikit-learn)='
```

Создать `requirements.txt` рядом с ботом:

```
vk_api==11.10.0
python-dotenv==1.0.1
gigachat==0.1.38
pydantic==2.9.2
scikit-learn==1.5.2
```

> Версии подставь **свои**, из вывода команды выше.
> `numpy`/`scipy` вписывать не надо — приедут с `scikit-learn`. `requests` тоже не надо — приедет с `vk_api`.
> `pydantic` обязательно 2.x: в коде `model_validate_json` и `min_length`, в pydantic 1 этого нет.

Проверка, что список полный: создай чистый venv локально, поставь из файла, запусти бота этим venv'ом.

```bash
python3 -m venv /tmp/check && /tmp/check/bin/pip install -r requirements.txt
/tmp/check/bin/python bot_terminal_vk.py
```

### 0.2. .gitignore

```
.env
.venv/
bot.log
__pycache__/
```

`.env` в репозиторий не попадает никогда — токены ВК и GigaChat из публичного репозитория утекают за часы.

### 0.3. Правки в коде под долгую жизнь

На десктопе ты перезапускал бота руками, на сервере он живёт месяцами. Что стоит поправить:

| Проблема | Что сделать |
|---|---|
| `CONVERSATIONS` растёт бесконечно | ограничить число `peer_id` или чистить по времени последнего сообщения |
| `verify_ssl_certs=False` | поставить сертификат Минцифры (шаг 7) и вернуть `True` |
| `bot.log` растёт бесконечно | logrotate (шаг 10) |
| `print` вместо логов | со временем перевести всё на `logging` |

Минимальная защита от разрастания истории:

```python
MAX_DIALOGS = 200

def build_messages(peer_id, user_text):
    if peer_id not in CONVERSATIONS and len(CONVERSATIONS) >= MAX_DIALOGS:
        CONVERSATIONS.pop(next(iter(CONVERSATIONS)))   # выбрасываем самый старый
    ...
```

---

## Шаг 1. Создаём сервер в Timeweb Cloud

Панель → **Облачные серверы** → *Создать*.

| Параметр | Значение | Почему |
|---|---|---|
| Регион | Россия (Москва / СПб) | GigaChat режет доступ с зарубежных IP |
| ОС | Ubuntu 24.04 LTS | свежая, все пакеты в наличии |
| Конфигурация | 1 CPU / 1–2 GB RAM / 15 GB NVMe | самый младший тариф; sklearn при импорте ест ~150–200 МБ |
| Сеть | **обязательно публичный IPv4** | см. ниже |
| SSH-ключ | добавить свой | удобнее и безопаснее пароля |

### Про IPv4 — важно

Некоторые дешёвые тарифы идут **IPv6-only**. С таким сервером:

- ты не зайдёшь по SSH, если у твоего домашнего провайдера нет IPv6 (`ping6 google.com` → `No route to host`);
- и главное — **бот вообще не заработает**: VK API и GigaChat по IPv6 недоступны.

Если IPv4 уже нет — докупи в панели (обычно 100–200 ₽/мес) или смени тариф. Проверить на сервере:

```bash
ip -4 addr        # должен быть адрес вида x.x.x.x, кроме 127.0.0.1
```

Если по SSH зайти не получается — в панели Timeweb есть **веб-консоль** (VNC), ей IPv6/IPv4 клиента не важен.

Пароль root придёт на почту или будет в панели.

---

## Шаг 2. Первый вход

```bash
ssh root@ТВОЙ_IP
```

macOS/Linux — из терминала. Windows — PowerShell (`ssh` встроен) или PuTTY.

При первом подключении спросит про отпечаток ключа — отвечай `yes`.

Обновляем систему и ставим базовое:

```bash
apt update && apt upgrade -y
apt install -y python3-venv python3-pip git curl logrotate
timedatectl set-timezone Europe/Moscow
```

---

## Шаг 3. Проверяем сеть — до всего остального

Это важнее, чем кажется: у бота три внешних зависимости, и любая может оказаться недоступной именно с этого сервера.

```bash
curl -s -o /dev/null -w "vk.ru:   %{http_code}\n" -m 10 https://api.vk.ru/method/utils.getServerTime
curl -s -o /dev/null -w "lp.vk:   %{http_code}\n" -m 10 https://lp.vk.com/
curl -s -o /dev/null -w "giga:    %{http_code}\n" -m 10 https://api.giga.chat/
```

Любой HTTP-код (`200`, `401`, `404`) = связь есть. **Таймаут или `000` = проблема.**

Почему проверяем `lp.vk.com` отдельно: патч `use_vk_ru()` переписывает только `vk_session.http.post`, а `VkBotLongPoll` ходит GET-ом на адрес, который вернул сам ВК — обычно это `lp.vk.com`. Метод `groups.getLongPollServer` может пройти, а `listen()` — молча повиснуть.

Полная проверка long-poll (подставь свои токен и ID группы):

```bash
python3 - <<'EOF'
import requests
TOKEN, GID = "ТОКЕН_ГРУППЫ", 240322029
r = requests.get("https://api.vk.ru/method/groups.getLongPollServer",
                 params={"group_id": GID, "access_token": TOKEN, "v": "5.131"},
                 timeout=10).json()
print(r)
s = r["response"]
print("server:", s["server"])
print(requests.get(s["server"], params={"act": "a_check", "key": s["key"],
                   "ts": s["ts"], "wait": 5}, timeout=15).text)
EOF
```

Видишь `{"ts":"...","updates":[]}` — всё в порядке, можно продолжать.

---

## Шаг 4. Пользователь, swap, SSH-ключи, файрвол

### 4.1. Отдельный пользователь

Бот не должен работать от root: любая дыра в коде — сразу полный доступ к серверу.

```bash
adduser --disabled-password --gecos "" bot
```

> `--disabled-password` значит, что у `bot` нет пароля и он **не может** делать `sudo` — при попытке получишь `sudo: I'm sorry bot. I'm afraid I can't do that`. Это нормально и задумано: всё администрирование делаем из root-сессии.
>
> Если хочешь дать ему права: `usermod -aG sudo bot` + `passwd bot` (из-под root), затем переподключиться.

### 4.2. Swap

Если RAM 1 ГБ — обязательно, иначе `pip install scikit-learn` может убиться по OOM прямо на сборке.

```bash
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
free -h        # проверка
```

### 4.3. Вход по ключу вместо пароля

С **локальной** машины, в отдельном терминале:

```bash
ssh-copy-id root@ТВОЙ_IP
```

Нет ключа — сначала `ssh-keygen -t ed25519`.

Затем на сервере в `/etc/ssh/sshd_config`:

```
PasswordAuthentication no
PermitRootLogin prohibit-password
```

```bash
systemctl restart ssh
```

> **Не закрывай текущую сессию**, пока не проверишь вход по ключу из нового окна. Иначе останется только веб-консоль.

### 4.4. Файрвол

```bash
ufw allow OpenSSH
ufw --force enable
ufw status
```

Входящих портов боту не нужно — только исходящие соединения.

---

## Шаг 5. Заливаем код

### Вариант А — через git

```bash
su - bot
git clone https://github.com/USER/REPO.git vkbot
cd vkbot
```

Приватный репозиторий — либо deploy key, либо HTTPS с токеном.

### Вариант Б — через scp с локальной машины

```bash
scp -r ./LMSH root@ТВОЙ_IP:/home/bot/vkbot
ssh root@ТВОЙ_IP "chown -R bot:bot /home/bot/vkbot"
```

Проверь, что `corpus.txt` доехал — без него бот падает на старте с `RuntimeError: corpus.txt пустой`.

```bash
ls -la /home/bot/vkbot/
```

---

## Шаг 6. Виртуальное окружение

От пользователя `bot`:

```bash
su - bot
cd ~/vkbot
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Установка `scikit-learn` на слабом сервере занимает пару минут — это нормально.

---

## Шаг 7. Секреты и сертификат

### 7.1. .env

```bash
nano ~/vkbot/.env
```

```
VK_API=vk1.a.ТВОЙ_ТОКЕН_СООБЩЕСТВА
GROUP_ID=240322029
GIGACHAT_CREDENTIALS=ТВОЙ_КЛЮЧ_АВТОРИЗАЦИИ
```

```bash
chmod 600 ~/vkbot/.env
```

`load_dotenv()` найдёт файл сам, потому что в юните задан `WorkingDirectory`.

### 7.2. Сертификат Минцифры (чтобы убрать verify_ssl_certs=False)

GigaChat использует сертификат российского УЦ, которого нет в системном хранилище Ubuntu. Отсюда и костыль с отключённой проверкой. Правильное решение — под **root**:

```bash
curl -k -o /usr/local/share/ca-certificates/russian_trusted_root_ca.crt \
  https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt
update-ca-certificates
```

После этого в коде:

```python
verify_ssl_certs=True
```

Не заработало — верни `False` и вернись к этому позже, на работу бота это не влияет.

---

## Шаг 8. Тестовый запуск руками

**Обязательный шаг.** Не заворачивай в systemd то, что ещё ни разу не стартовало — отлаживать через `journalctl` дольше.

```bash
su - bot
cd ~/vkbot
.venv/bin/python -u bot_terminal_vk.py
```

Ожидаемый вывод:

```
🤖 Бот VK + GigaChat запущен!
📊 Группа ID: 240322029
🌐 VK API: https://api.vk.ru/
🤖 Модель GigaChat: GigaChat-2
📚 Корпус: N кусков
Ожидание сообщений...
```

Напиши боту в ВК **с личной страницы** (не от имени сообщества — такие сообщения отсекает `if from_id < 0: return`). Должен ответить.

Тишина в ответ — проверь настройки сообщества: Управление → Работа с API → Long Poll API включён, версия 5.131, во вкладке «Типы событий» отмечено «Входящее сообщение» (по умолчанию галка снята). И Сообщения → Сообщения сообщества включены.

Остановить: `Ctrl+C`.

---

## Шаг 9. systemd — служба

### Зачем

Пока бот запущен из терминала: закрыл SSH — умер, упал с ошибкой — никто не поднял, перезагрузился сервер — тишина. systemd решает всё три проблемы.

### Юнит

Файл создаётся **из-под root** (у `bot` нет прав на `/etc`):

```bash
nano /etc/systemd/system/vkbot.service
```

```ini
[Unit]
Description=VK GigaChat bot
After=network-online.target
Wants=network-online.target

[Service]
User=bot
Group=bot
WorkingDirectory=/home/bot/vkbot
ExecStart=/home/bot/vkbot/.venv/bin/python -u bot_terminal_vk.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# немного изоляции
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Разбор:

| Строка | Что делает |
|---|---|
| `User=bot` | запуск от непривилегированного пользователя |
| `WorkingDirectory=` | отсюда находятся `.env` и `corpus.txt` |
| `ExecStart=` | полный путь к python **из venv**, не системный |
| `-u` | без него `print` копится в буфере и журнал пустой |
| `Restart=always` | упал — поднять |
| `RestartSec=5` | пауза перед перезапуском, чтобы не долбить в цикле |
| `WantedBy=multi-user.target` | автозапуск при загрузке сервера |

> Если файл писался под `bot` и не сохранился — сохрани в домашнюю папку (Ctrl+O, путь `/home/bot/vkbot.service`), потом из root:
> `install -m 644 -o root -g root /home/bot/vkbot.service /etc/systemd/system/`

### Запуск

```bash
systemctl daemon-reload          # перечитать файлы юнитов
systemctl enable --now vkbot     # enable = автозапуск, --now = запустить сейчас
systemctl status vkbot           # проверить
```

Должно быть `Active: active (running)`.

### Команды на каждый день

| Команда | Что делает |
|---|---|
| `systemctl status vkbot` | состояние + последние строки вывода |
| `systemctl restart vkbot` | перезапустить (после изменения кода) |
| `systemctl stop vkbot` | остановить |
| `systemctl start vkbot` | запустить |
| `systemctl disable vkbot` | убрать из автозапуска |
| `journalctl -u vkbot -f` | живой лог, выход по Ctrl+C |
| `journalctl -u vkbot -n 100` | последние 100 строк |
| `journalctl -u vkbot --since "10 min ago"` | за последние 10 минут |
| `journalctl -u vkbot -p err` | только ошибки |

`-u` = unit (служба), `-f` = follow (дописывать новые строки в реальном времени, как `tail -f`).

---

## Шаг 10. Ротация логов

`bot.log` только растёт. За полгода он может занять гигабайты и забить диск, а забитый диск роняет всё подряд.

**Ротация** — регулярная нарезка: раз в неделю текущий файл уходит в архив (`bot.log.1.gz`), запись начинается с чистого, архивы старше 4 штук удаляются. `logrotate` уже стоит в Ubuntu и запускается сам раз в сутки.

Под root:

```bash
nano /etc/logrotate.d/vkbot
```

```
/home/bot/vkbot/bot.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su bot bot
}
```

`copytruncate` — ключевая строка. Обычно logrotate переименовывает файл и создаёт новый, но Python держит открытым старый дескриптор и продолжает писать в переименованный — новый останется пустым навсегда. `copytruncate` копирует содержимое в архив и обнуляет файл на месте, не трогая дескриптор.

Проверка без последствий:

```bash
logrotate -d /etc/logrotate.d/vkbot     # режим отладки, только показывает план
```

Журнал systemd ограничивается отдельно, в `/etc/systemd/journald.conf`:

```
SystemMaxUse=200M
```

```bash
systemctl restart systemd-journald
```

---

## Шаг 11. Обновление кода

```bash
su - bot
cd ~/vkbot
git pull                                    # или scp с локальной машины
.venv/bin/pip install -r requirements.txt   # если зависимости менялись
exit
systemctl restart vkbot                     # из-под root
systemctl status vkbot
```

Правил юнит (`.service`) — обязательно `systemctl daemon-reload` перед `restart`, иначе systemd работает по старой версии файла.

---

## Траблшутинг

| Симптом | Причина | Лечение |
|---|---|---|
| `status=203/EXEC` | неверный путь в `ExecStart` | `ls /home/bot/vkbot/` — сверь имя файла и путь к python в venv |
| `status=200/CHDIR` | нет `WorkingDirectory` | проверь путь, права `chown -R bot:bot` |
| `ModuleNotFoundError` | взят системный python, не из venv | в `ExecStart` полный путь `/home/bot/vkbot/.venv/bin/python` |
| `RuntimeError: corpus.txt пустой` | файл не доехал | скопируй `corpus.txt` в `/home/bot/vkbot/` |
| Служба в цикле рестартов | падает на старте | `journalctl -u vkbot -n 50` — там трейсбек |
| В журнале пусто, бот работает | нет `-u` в `ExecStart` | добавь, `daemon-reload`, `restart` |
| Стартует, но не отвечает в ВК | нет доступа к `lp.vk.com` **или** выключены типы событий в сообществе | проверки из шага 3 + настройки Long Poll API |
| Половина сообщений теряется | запущено два экземпляра | `ps aux \| grep bot_terminal` — старый процесс из ручного запуска |
| Ошибки SSL к GigaChat | нет сертификата Минцифры | шаг 7.2 или временно `verify_ssl_certs=False` |
| `sudo: I'm sorry bot...` | `bot` не в sudoers | работай из root-сессии (так и задумано) |
| Убивается по памяти при `pip install` | нет swap | шаг 4.2 |

---

## Чек-лист перед тем, как оставить бота работать

- [ ] `curl` до `api.vk.ru`, `lp.vk.com`, `api.giga.chat` отвечает
- [ ] `.env` на месте, `chmod 600`, в `.gitignore`
- [ ] `corpus.txt` на сервере
- [ ] бот отвечает при запуске руками
- [ ] `systemctl status vkbot` → `active (running)`
- [ ] `systemctl enable vkbot` сделан (проверка: `systemctl is-enabled vkbot`)
- [ ] после `reboot` бот поднялся сам
- [ ] `journalctl -u vkbot -f` показывает живой вывод
- [ ] logrotate настроен, `logrotate -d` без ошибок
- [ ] вход по SSH-ключу работает, пароль отключён
- [ ] `ufw status` → active

Финальная проверка целиком:

```bash
reboot
# подождать минуту
ssh root@ТВОЙ_IP
systemctl status vkbot
```

Написал боту в ВК, он ответил — деплой закончен.
