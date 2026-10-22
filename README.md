# DataLab — узбекский речевой датасет

MP4 → wav 16 kHz mono → транскрипция **Xonsaroy STT** (или VoiceLab) → сегменты 1.5–15 с → SQLite + файлы
→ проверка и правка текста каждого сегмента в веб-интерфейсе.

## Какой STT используется

`STT_PROVIDER` в `.env` (или флаг `--provider`):

- **`xonsaroy`** (по умолчанию). API отдаёт только сплошной текст, без таймкодов и спикеров. Поэтому людей локально
  делит **pyannote/speaker-diarization-3.1** (см. «Диаризация спикеров» ниже), реплика каждого спикера режется на куски
  1.5–15 с по паузам, и каждый кусок распознаётся отдельно с меткой своего спикера. Куски разных спикеров, которые
  пересекаются по времени, получают статус `overlap`. Без pyannote (не установлен, нет `HF_TOKEN`) или с
  `--no-diarization` всё видео режется только по паузам. Пауза длиннее 1.5 с
  всегда граница куска; если без разреза кусок длиннее 15 с, он режется в самой длинной паузе внутри, чтобы фраза
  не делилась пополам
  (`POST /api/v1/transcriptions`, не больше 2 запросов одновременно, повтор только на 429/502/503 и обрыв связи).
  Таймкод сегмента = границы куска. Без диаризации все сегменты видео получают `SPEAKER_00`, и разных людей разводит
  редактор в студии (переназначение спикера, Alt+1..9). Если упало посередине, «↻ Повторить» продолжит с первого
  нераспознанного куска (`transcripts/<video>.xonsaroy.partial.json`). Ключ: `XONSAROY_API_KEY=xs_stt_...`.
- **`voicelab`**. Весь файл целиком, VoiceLab сам режет на реплики и делит спикеров, тратит кредиты.
  Ключ: `VOICELAB_API_KEY=vlk_...`.

Уже готовые расшифровки (`transcripts/<video>.voicelab.json`) при смене сервиса не переделываются; чтобы прогнать
видео через другой сервис, удалите этот файл или запустите с `--force`.

## Установка

Нужны Python 3.10+ и [ffmpeg](https://ffmpeg.org/download.html) в `PATH`.

```bash
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
# в .env впишите XONSAROY_API_KEY=xs_stt_... (или STT_PROVIDER=voicelab и VOICELAB_API_KEY)
```

### Диаризация спикеров (pyannote, локально)

pyannote работает на вашем компьютере, в Xonsaroy уходит только текст каждой реплики. Видеокарта NVIDIA ускоряет
в разы (GTX 1650 4 GB хватает), без неё работает на процессоре.

```bash
# torch с CUDA (GTX 1650 подходит, нужен драйвер NVIDIA 528+); без видеокарты: --index-url https://download.pytorch.org/whl/cpu
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-diarization.txt
python -c "import torch; print(torch.cuda.is_available())"   # True = будет cuda
```

1. Токен Hugging Face (Read): https://hf.co/settings/tokens, впишите в `.env`: `HF_TOKEN=hf_...`.
2. Тем же аккаунтом примите условия на https://hf.co/pyannote/speaker-diarization-3.1 и
   https://hf.co/pyannote/segmentation-3.0. Модель скачивается один раз при первом запуске.
3. Устройство выбирается само (`cuda`, если есть, иначе `cpu`); вручную: `--device cpu` или `DIARIZATION_DEVICE=cpu`
   в `.env`. Если памяти видеокарты не хватило, видео повторяется на cpu.

Реплики спикеров кэшируются в `transcripts/<video>.diarization.json`. Если pyannote не загрузился, в логе будет
«ВНИМАНИЕ: диаризация недоступна» с причиной, а видео распознается одним спикером; когда pyannote заработает,
перезапустите его с `--force` (или удалите `transcripts/<video>.voicelab.json`). В Docker всё это уже внутри образа,
см. «Диаризация в Docker» ниже.

## Запуск без Docker

```bash
uvicorn app.server:app --port 8000   # веб-интерфейс
python -m pipeline.prepare           # или пакетно: видео из data/videos/
```

Флаги: `--provider xonsaroy|voicelab`, `--no-diarization`, `--device auto|cuda|cpu`, `--num-speakers N` (если число
людей в видео известно), `--videos`, `--out`, `--language uz`, `--min-dur 1.5`, `--max-dur 15`, `--max-gap 0.5`,
`--keep-overlap` (оставить наложения со статусом `overlap`), `--no-speakers` (только voicelab),
`--upload-format wav|flac|mp3` (только voicelab; если большой файл не принимается, попробуйте `mp3`), `--force`, `--allow-duplicates`.

### Только разделить спикеров (без STT)

Если нужно просто разделить людей в видео, без распознавания текста, ключей API и кредитов:

```bash
python -m pipeline.split_speakers --num-speakers 2          # все видео из data/videos/
python -m pipeline.split_speakers --videos film.mp4         # один файл
```

Работает только pyannote (нужны `HF_TOKEN` и `requirements-diarization.txt`). Результат в `data/speakers/<video>/`:
папки `SPEAKER_00/`, `SPEAKER_01/` с репликами 1.5–15 с, `SPEAKER_00.wav` (вся речь спикера одним файлом),
`speakers.csv` (таймкоды) и `summary.json` (секунды на спикера). Участки, где говорят одновременно, вырезаются
из реплик; `--keep-overlap` сохраняет их в `overlap/`. Кэш диаризации общий с `prepare.py`, поэтому потом
`prepare` не будет диаризовать то же видео заново. Флаги: `--device`, `--min-dur`, `--max-dur`, `--max-gap`, `--force`.

Результат STT кэшируется в `transcripts/<video>.voicelab.json` (имя историческое, поле `provider` говорит, кто его
сделал), поэтому повторный запуск не отправляет видео заново (кроме `--force`).

Защита от дублей, чтобы не отправлять в STT дважды:
- **Тот же файл** (совпадает sha256) веб-интерфейс не принимает и сообщает, под каким именем он уже загружен.
- **Тот же звук в другом файле** (другое имя, MKV вместо MP4, перекодированное видео) определяется по отпечатку
  громкости (`transcripts/audio_index.json`). Такое видео пропускается со статусом «дубликат», в VoiceLab не отправляется
  и не попадает в датасет второй раз. Если дубль всё же нужен, используйте `--allow-duplicates`: расшифровка возьмётся из кэша.
- **Обрыв связи с VoiceLab** (`IncompleteRead`, `Connection broken`) повторяется автоматически. Если задача всё же
  упала, нажмите «↻ Повторить» у неё или загрузите тот же файл ещё раз: продолжится та же расшифровка VoiceLab
  (тот же Idempotency-Key из `transcripts/<video>.pending.json`), кредиты второй раз не спишутся.

## Веб-интерфейс (Docker)

Нужен только Docker (Python, ffmpeg, torch и pyannote уже внутри образа).

```bash
cp .env.example .env              # Windows: copy .env.example .env, впишите ключ
docker compose up -d --build      # открыть http://localhost:8000
```

### Диаризация в Docker (pyannote на видеокарте)

Образ собирается с torch 2.6 (CUDA 12.4) и pyannote: первая сборка долгая, образ занимает около 10 ГБ на диске. В `.env` нужен
`HF_TOKEN=hf_...` (и принятые условия моделей, см. «Диаризация спикеров»). Модель скачивается при первом видео и
хранится в `data/.cache/huggingface`.

Видеокарту контейнеру даёт `docker-compose.override.yml` (compose подхватывает его сам). На Windows нужны свежий
драйвер NVIDIA и Docker Desktop на WSL2. Проверка: `docker compose exec web python -c "import torch; print(torch.cuda.is_available())"`
должно напечатать `True`, а в логе задачи будет `pyannote: ... на cuda`.

- Docker пишет `could not select device driver "nvidia"`: видеокарта контейнеру недоступна. Удалите или переименуйте
  `docker-compose.override.yml`, pyannote будет работать на cpu (медленнее).
- Образ без pyannote (меньше, нарезка только по паузам): `DIARIZATION=0 docker compose up -d --build`.

### Доступ из интернета (Cloudflare Tunnel)

Туннель открывает студию по адресу `https://...` без белого IP и проброса портов. Сначала задайте пароль в `.env`:

```
STUDIO_USER=admin
STUDIO_PASSWORD=придумайте-длинный-пароль
```

Без `STUDIO_PASSWORD` студия работает только на этом компьютере: запросы через туннель получают 403.
С паролем браузер спрашивает логин и пароль и на `localhost`.

**Постоянный адрес на своём домене** (домен должен быть в Cloudflare):

1. [one.dash.cloudflare.com](https://one.dash.cloudflare.com) → Networks → Tunnels → Create a tunnel → Cloudflared, имя `speech-lab`.
2. На шаге установки скопируйте токен (длинная строка после `--token`) и впишите в `.env`: `CLOUDFLARE_TUNNEL_TOKEN=eyJ...`.
   Токен храните только в `.env`, в чат и в git его не вставляйте.
3. Public Hostname: например `lab.ваш-домен.uz`, Service `HTTP`, URL `web:8000`.
4. Запуск: `docker compose --profile tunnel up -d --build`, проверка: `docker compose logs tunnel` (строки `Registered tunnel connection`).

**Временный адрес без домена:** `docker compose --profile quick up -d --build`, затем `docker compose logs quicktunnel`
покажет адрес `https://<слова>.trycloudflare.com`. Он меняется при каждом перезапуске контейнера.

Остановить туннель: `docker compose --profile tunnel down` (или `--profile quick`).
Cloudflare на бесплатном тарифе принимает загрузку файла не больше 100 МБ: большие видео загружайте через `localhost`
или положите в `data/videos` и запустите `docker compose run --rm prepare`.

Студия устроена как монтажка (DaVinci Resolve):

- **Медиа** (слева): перетащите видео или аудио. Файл встаёт в очередь, VoiceLab его расшифровывает, видео появляется в списке.
- **Просмотр** (в центре): исходное видео, поверх него субтитром идёт текст реплики под плейхедом. Сразу видно, совпадает ли текст со звуком.
- **Таймлайн** (внизу): волна звука и отдельная дорожка на каждого спикера с клипами-сегментами. Клик по линейке перематывает,
  `Ctrl`+колёсико меняет масштаб, края клипа тянутся мышкой (новые границы сегмента).
- **Инспектор** (справа): спикер, вход/выход (кнопки `I ⟵`/`O ⟵` берут время из плейхеда), «▶ Сегмент», «⟳ Повтор», текст,
  кнопки `oʻ gʻ ʼ`, «Исправить апострофы», предупреждения о цифрах и кириллице, «Одобрить», «Брак», «Наложение», «Сохранить».
- **Спикеры**: метка `SPEAKER_xx` от VoiceLab остаётся, редактор только уточняет, кто говорит.
  Двойной щелчок по имени дорожки (или `✎`) даёт спикеру имя, `⇄` переносит все его реплики к другому спикеру
  (если VoiceLab разделил одного человека на двоих), «+ Спикер» добавляет дорожку. Клип можно перетащить на дорожку
  другого спикера, выбрать спикера в инспекторе или нажать `Alt+1…9`. Имя попадает в ZIP (`speaker_name`) и в тексты SRT/TXT (`Ali: …` вместо `SPEAKER_00: …`). Сегменты со статусом «брак» в ZIP не попадают.

**Добавить и удалить сегмент.** «+ Текст» (или `N`) создаёт сегмент на 3 с от плейхеда, двойной щелчок по пустому месту
дорожки создаёт его у этого спикера. Границы подгоняются `I`/`O` или краями клипа, wav режется из полного аудио.
🗑 в инспекторе (или `Delete`) удаляет сегмент вместе с wav; во всплывающем сообщении есть «Вернуть».

**Очистить** (вверху справа) убирает все видео, аудио, сегменты, тексты и архивы в корзину `data/trash/<дата>/`.
Сразу после очистки можно нажать «Вернуть», а позже открыть **Корзину** и вернуть любую очистку со всеми правками
и именами спикеров. «Удалить» в корзине стирает очистку насовсем. Пока идёт обработка видео, очистка недоступна.

Клавиши: `Space` пуск/пауза, `J`/`K`/`L`, `←`/`→` предыдущий/следующий клип, `I`/`O` вход/выход, `Enter` проиграть сегмент, `Alt+1…9` спикер клипа,
`Ctrl+Enter` одобрить, `Ctrl+Shift+X` брак, `Ctrl+Shift+O` наложение, `Ctrl+S` сохранить, `Ctrl+↓` следующий непроверенный.
Простой список сегментов: http://localhost:8000/list.html.

**Скачать всё одним ZIP.** Кнопка «⬇ ZIP» у видео, «⬇ ZIP всё» и «⬇ ZIP одобренные» в шапке. Внутри архива:
`<video>/video.mp4` (исходник), `<video>/audio.wav`, `<video>/segments/*.wav`, `<video>/transcript.txt` и `.srt`
(с правками разметчиков), `<video>/segments.jsonl`, `<video>/voicelab.json`, плюс общие `manifest.jsonl` и `manifest.csv`.
Копия архива остаётся в `data/exports/`. Без браузера: `python -m app.bundle [--status approved] [--video NAME]`
(в Docker: `docker compose run --rm --entrypoint python prepare -m app.bundle`). «Экспорт approved.jsonl» отдаёт проверенные сегменты.

**Расход STT.** Кнопка «STT: N мин» в шапке показывает, сколько аудио отправлено (и кредитов VoiceLab, если он
использовался). Xonsaroy кредитов не списывает, для него пишутся минуты отправленной речи. По щелчку открываются итоги,
расход по дням и журнал: какое видео, сколько минут аудио и кредитов, какие дубли не отправлялись. Кредиты берутся из
`GET /v1/stt/preflight` по фактической длительности файла (так VoiceLab и списывает; запрос бесплатный). Упавшие задачи
записываются с 0 кредитов, потому что VoiceLab возвращает их резерв. Журнал лежит в `data/usage.jsonl`, кнопка «Очистить»
его не трогает. Баланс счёта API VoiceLab не отдаёт, его смотрите в кабинете VoiceLab. Пакетный запуск в конце печатает
расход за запуск и за всё время. API: `GET /api/usage`.

**Только текст: SRT и TXT.** В меню «⬇ Экспорт»: «SRT всех видео», «TXT всех видео», «SRT + TXT всех видео»
(ZIP с файлом `<video>.srt` / `<video>.txt` на каждое видео) и «SRT/TXT этого видео» для открытого видео.
В файлах текст с правками разметчиков и спикеры с именами, которые дал редактор (без имени остаётся `SPEAKER_00`).
Брак и пустые клипы не попадают. Копии лежат в `data/exports/transcripts/`.
API: `/api/transcripts.zip?fmt=srt|txt|all[&status=approved]`, `/api/transcript/<video>.srt|.txt`.

Пакетная обработка папки `data/videos` без браузера:

```bash
docker compose run --rm prepare   # флаги передаются как есть, например --upload-format mp3
```

## Что получается в `data/`

| Путь | Содержимое |
|---|---|
| `audio/<video>.wav` | полное аудио, 16 kHz mono 16-bit PCM |
| `segments/<video>/<video>_<start_ms>_<end_ms>.wav` | реплики одного спикера 1.5–15 с |
| `transcripts/<video>.txt` / `.srt` | полная расшифровка со спикерами и таймкодами |
| `transcripts/<video>.voicelab.json` | сегменты от STT (Xonsaroy или VoiceLab) |
| `transcripts/<video>.diarization.json` | реплики спикеров от pyannote (только Xonsaroy) |
| `manifest.jsonl` | все сегменты: `audio_filepath`, `duration`, `text`, `speaker`, `video`, `status` |
| `speech_lab.db` | SQLite, таблица `segments` (схема speech-lab) |
| `usage.jsonl` | журнал расхода STT: отправки, минуты, кредиты VoiceLab, пропущенные дубли |
| `exports/*.zip` | архивы «видео + аудио + сегменты + текст», собранные кнопкой ZIP |
| `exports/transcripts/<video>.srt` / `.txt` | последние скачанные тексты с правками и именами спикеров |

`speaker` = `<video>_SPEAKER_xx`, потому что STT нумерует спикеров заново в каждом видео.
