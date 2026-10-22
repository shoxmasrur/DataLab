"""Видео -> спикеры по отдельности, без STT: только pyannote на своём компьютере, ключи API и кредиты не нужны.

    python -m pipeline.split_speakers                        # все видео из data/videos/
    python -m pipeline.split_speakers --videos film.mp4 --num-speakers 2

Что получается в --out (по умолчанию data/):
    audio/<video>.wav                                   полное аудио, 16 kHz mono 16-bit PCM
    transcripts/<video>.diarization.json                реплики pyannote (общий кэш с prepare.py)
    speakers/<video>/SPEAKER_00/<video>_<ms>_<ms>.wav   реплики спикера, 1.5–15 с
    speakers/<video>/SPEAKER_00.wav                     вся речь спикера одним файлом (паузы 0.3 с)
    speakers/<video>/speakers.csv                       file, speaker, start, end, duration, overlap
    speakers/<video>/summary.json                       сколько секунд и реплик у каждого спикера

Наложения (два человека говорят одновременно) по умолчанию выбрасываются; --keep-overlap кладёт их
в папку overlap/. В SQLite ничего не пишется: это просто разделение, распознавание делает prepare.py.
"""
import argparse
import csv
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

from pipeline import diarize  # noqa: E402  (единственная зависимость внутри проекта)

# Скрипт не импортирует prepare.py, чтобы работать с любой его версией.
MEDIA_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4a", ".mp3", ".wav", ".ogg", ".flac"}
SAMPLE_RATE = 16000
PAD = 0.05          # запас по краям реплики, с
OVERLAP_MIN = 0.1   # пересечение с другим спикером, после которого участок считается наложением, с
JOIN_SILENCE = 0.3  # пауза между репликами в общем файле спикера, с


class MediaError(RuntimeError):
    pass


def load_env(path: str | Path) -> None:
    """KEY=VALUE из .env в os.environ (HF_TOKEN, DIARIZATION_DEVICE)."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def safe_name(stem: str) -> str:
    return re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_") or "video"


def to_wav(src: Path, dst: Path) -> None:
    """ffmpeg -> wav 16 kHz mono 16-bit."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
                        "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dst)],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        dst.unlink(missing_ok=True)
        why = " ".join(r.stderr.strip().splitlines()[-3:]) or f"код {r.returncode}"
        raise MediaError(f"ffmpeg не смог прочитать {src.name}: {why}")


def cut_wav(src: Path, dst: Path, start: float, end: float) -> None:
    with wave.open(str(src), "rb") as r:
        sr = r.getframerate()
        r.setpos(int(start * sr))
        frames = r.readframes(int((end - start) * sr))
        params = r.getparams()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)


def get_turns(wav: Path, cache: Path, args) -> list[dict] | None:
    """Реплики pyannote; кэш transcripts/<video>.diarization.json (тот же, что у prepare.py)."""
    args.diarization_error = None
    path = cache.parent / f"{wav.stem}.diarization.json"
    if path.exists() and not args.force:
        return json.loads(path.read_text(encoding="utf-8"))["turns"]
    try:
        print(f"  pyannote: диаризация на {diarize.pick_device(args.device)} ...")
        try:
            turns = diarize.diarize(wav, device=args.device, num_speakers=args.num_speakers)
        except RuntimeError as e:  # не хватило памяти видеокарты: повтор на cpu
            if "out of memory" not in str(e).lower() or args.device == "cpu":
                raise
            print("  pyannote: не хватило памяти видеокарты, повтор на cpu")
            diarize.torch_empty_cache()
            turns = diarize.diarize(wav, device="cpu", num_speakers=args.num_speakers)
    except Exception as e:  # noqa: BLE001
        args.diarization_error = str(e)[:300]
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": diarize.MODEL, "turns": turns}, ensure_ascii=False, indent=1), encoding="utf-8")
    return turns


def clean_turns(turns: list[dict]) -> list[dict]:
    """Вырезает из реплик участки, где одновременно говорит другой спикер (дольше OVERLAP_MIN).
    Чистые части остаются у спикера (overlap=False), вырезанные участки идут отдельно (overlap=True),
    поэтому секундное наложение в конце не выбрасывает всю 20-секундную реплику."""
    out = []
    for t in turns:
        cuts = sorted((max(t["start"], o["start"]), min(t["end"], o["end"])) for o in turns
                      if o["speaker"] != t["speaker"] and min(t["end"], o["end"]) - max(t["start"], o["start"]) > OVERLAP_MIN)
        pos = t["start"]
        for a, b in cuts:
            if a > pos:
                out.append({**t, "start": pos, "end": a, "overlap": False})
            if b > pos:
                out.append({**t, "start": max(a, pos), "end": b, "overlap": True})
                pos = b
        if t["end"] > pos:
            out.append({**t, "start": pos, "end": t["end"], "overlap": False})
    seen, uniq = set(), []  # наложение A и B попадает и из реплики A, и из реплики B: оставляем одно
    for x in sorted(out, key=lambda x: (x["start"], x["end"])):
        key = (round(x["start"], 2), round(x["end"], 2)) if x["overlap"] else id(x)
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq


def merge_turns(turns: list[dict], max_dur: float, max_gap: float) -> list[dict]:
    """Склеивает подряд идущие реплики одного спикера (пауза <= max_gap), пока кусок не длиннее max_dur."""
    out: list[dict] = []
    for t in turns:
        prev = next((p for p in reversed(out) if p["speaker"] == t["speaker"]), None)
        if (prev is not None and prev is out[-1] and not prev["overlap"] and not t["overlap"]
                and t["start"] - prev["end"] <= max_gap and t["end"] - prev["start"] <= max_dur):
            prev["end"] = t["end"]
        else:
            out.append(dict(t))
    return out


def split_long(t: dict, max_dur: float) -> list[dict]:
    """Без слов границ нет, поэтому длинная реплика режется на равные части не длиннее max_dur."""
    dur = t["end"] - t["start"]
    if dur <= max_dur:
        return [t]
    n = int(dur // max_dur) + 1
    step = dur / n
    return [{**t, "start": t["start"] + i * step, "end": t["start"] + (i + 1) * step} for i in range(n)]


def join_speaker(src: Path, dst: Path, pieces: list[dict]) -> None:
    """Вся речь спикера в одном wav: реплики по времени, между ними JOIN_SILENCE тишины."""
    with wave.open(str(src), "rb") as r:
        params, sr = r.getparams(), r.getframerate()
        gap = b"\x00\x00" * int(JOIN_SILENCE * sr)
        with wave.open(str(dst), "wb") as w:
            w.setparams(params)
            for i, p in enumerate(pieces):
                r.setpos(int(p["start"] * sr))
                w.writeframes(r.readframes(int((p["end"] - p["start"]) * sr)))
                if i < len(pieces) - 1:
                    w.writeframes(gap)


def process(src: Path, out: Path, args) -> dict | None:
    name = safe_name(src.stem)
    wav = out / "audio" / f"{name}.wav"
    if not wav.exists() or args.force:
        print("  ffmpeg -> wav 16 kHz mono ...")
        to_wav(src, wav)
    with wave.open(str(wav), "rb") as r:
        total = r.getnframes() / r.getframerate()

    turns = get_turns(wav, out / "transcripts" / f"{name}.json", args)
    if turns is None:
        print(f"  ОШИБКА: диаризация не выполнена ({args.diarization_error}); видео пропущено", file=sys.stderr)
        return None

    turns = clean_turns(turns)
    pieces = [p for t in merge_turns(turns, args.max_dur, args.max_gap) for p in split_long(t, args.max_dur)]

    base = out / "speakers" / name
    rows, by_speaker = [], {}
    for p in pieces:
        if p["end"] - p["start"] < args.min_dur or (p["overlap"] and not args.keep_overlap):
            continue
        pad = 0.0 if p["overlap"] else PAD  # не захватывать соседнее наложение запасом по краям
        start, end = max(0.0, p["start"] - pad), min(total, p["end"] + pad)
        folder = "overlap" if p["overlap"] else p["speaker"]
        dst = base / folder / f"{name}_{int(start * 1000):07d}_{int(end * 1000):07d}.wav"
        cut_wav(wav, dst, start, end)
        rows.append({"file": dst.relative_to(base).as_posix(), "speaker": p["speaker"], "start": round(start, 3),
                     "end": round(end, 3), "duration": round(end - start, 3), "overlap": p["overlap"]})
        if not p["overlap"]:
            by_speaker.setdefault(p["speaker"], []).append({"start": start, "end": end})

    for spk, ps in by_speaker.items():
        join_speaker(wav, base / f"{spk}.wav", ps)

    with open(base / "speakers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "start", "end", "duration", "overlap"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "video": src.name, "duration": round(total, 1),
        "speakers": {spk: {"clips": len(ps), "seconds": round(sum(p["end"] - p["start"] for p in ps), 1)}
                     for spk, ps in sorted(by_speaker.items())},
        "overlap_clips": sum(r["overlap"] for r in rows),
    }
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for spk, s in summary["speakers"].items():
        print(f"  {spk}: {s['clips']} реплик, {s['seconds']} с")
    print(f"  -> {base}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Разделить речь в видео по спикерам (pyannote), без STT")
    ap.add_argument("--videos", default="data/videos", help="папка с видео (или один файл)")
    ap.add_argument("--out", default="data", help="куда сохранять результат")
    ap.add_argument("--env", default=str(ROOT / ".env"), help="файл с HF_TOKEN")
    ap.add_argument("--num-speakers", type=int, default=None, help="сколько людей в видео, если известно")
    ap.add_argument("--device", default=None, help="auto (cuda, если есть), cuda или cpu")
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--max-gap", type=float, default=0.5, help="макс. пауза при склейке реплик, с")
    ap.add_argument("--keep-overlap", action="store_true", help="сохранять наложения в папку overlap/")
    ap.add_argument("--force", action="store_true", help="заново конвертировать и диаризовать")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    load_env(args.env)
    src, out = Path(args.videos), Path(args.out)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in MEDIA_EXT)
    if not files:
        sys.exit(f"Нет видео в {src}")
    failed = 0
    for f in files:
        print(f"[{f.name}]")
        try:
            if process(f, out, args) is None:
                failed += 1
        except MediaError as e:
            print(f"  ОШИБКА: {e}", file=sys.stderr)
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
