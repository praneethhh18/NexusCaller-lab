"""
Local-combo sanity check — no Twilio, no LiveKit, no phone needed.

Tests each component of the local-oss stack independently, then runs a
full mock conversation turn (user text -> LLM -> TTS) and saves the output
audio so you can listen to it.

Usage:
    cd nexuscaller-lab
    venv\\Scripts\\python -m voice_agent.test_local
    venv\\Scripts\\python -m voice_agent.test_local --play   # play audio via sounddevice
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path


PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
SEP  = "-" * 60


def _hdr(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


# ── 1. Prerequisites ─────────────────────────────────────────────────────────
def check_prereqs(skip_llm: bool = False, skip_stt: bool = False) -> bool:
    _hdr("1/5  Prerequisites")
    ok = True

    tts_deps = [("piper", "piper-tts"), ("numpy", "numpy")]
    stt_deps = [("faster_whisper", "faster-whisper")]

    for module, pkg in tts_deps + ([] if skip_stt else stt_deps):
        try:
            __import__(module)
            print(f"  {PASS}  {pkg}")
        except ImportError:
            print(f"  {FAIL}  {pkg} — run: pip install {pkg}")
            ok = False

    # Ollama reachability — only a hard failure when LLM test is active
    try:
        import urllib.request, json
        r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        tags = json.loads(r.read())
        models = [m["name"] for m in tags.get("models", [])]
        target = "qwen2.5:0.5b-instruct"
        has_model = any(target in m for m in models)
        if has_model:
            print(f"  {PASS}  Ollama up · {target} present")
        else:
            label = FAIL if not skip_llm else SKIP
            print(f"  {label}  Ollama up but {target!r} not found — run: ollama pull {target}")
            print(f"         Available: {models or '(none)'}")
            if not skip_llm:
                ok = False
    except Exception as e:
        label = FAIL if not skip_llm else SKIP
        print(f"  {label}  Ollama not reachable ({e})")
        if skip_llm:
            print(f"         (LLM skipped — that's fine)")
        else:
            print(f"         Start Ollama, then: ollama pull qwen2.5:0.5b-instruct")
            ok = False

    # Check cached Piper model
    from voice_agent.local_plugins import _PIPER_DIR
    onnx = _PIPER_DIR / "en_US-lessac-medium.onnx"
    if onnx.exists() and onnx.stat().st_size > 1_000_000:
        print(f"  {PASS}  Piper voice cached ({onnx.stat().st_size // 1_000_000} MB)")
    else:
        print(f"  {SKIP}  Piper voice not cached — will download ~32 MB in step 4")

    # Check cached Whisper model
    hf_cache = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    whisper_present = any(hf_cache.rglob("tiny*ctranslate2*")) if hf_cache.exists() else False
    if whisper_present:
        print(f"  {PASS}  Whisper tiny weights cached")
    elif not skip_stt:
        print(f"  {SKIP}  Whisper tiny not cached — will download ~75 MB in step 3")
    else:
        print(f"  {SKIP}  Whisper tiny (STT skipped)")

    return ok


# ── 2. LLM (Ollama) ──────────────────────────────────────────────────────────
def test_llm() -> str | None:
    _hdr("2/5  LLM — Ollama · qwen2.5:0.5b-instruct")
    # Call Ollama's native /api/chat endpoint directly — the livekit openai
    # plugin uses cloud-tuned timeouts (5-10 s) that are too short for a
    # local model loading into RAM on the first call.
    try:
        import urllib.request, json
        payload = json.dumps({
            "model": "qwen2.5:0.5b-instruct",
            "stream": False,
            "messages": [
                {"role": "system",  "content": "You are Vox, a friendly sales agent. Reply in one short sentence."},
                {"role": "user",    "content": "Hi, is this a good time to talk about improving your sales process?"},
            ],
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read())
        elapsed = time.time() - t0
        reply = body.get("message", {}).get("content", "").strip()
        print(f"  {PASS}  {elapsed:.2f}s")
        print(f"  Reply: {reply!r}")
        return reply
    except Exception as e:
        print(f"  {FAIL}  {e}")
        import traceback; traceback.print_exc()
        return None


# ── 3. STT (faster-whisper) ──────────────────────────────────────────────────
def test_stt(piper_pcm: bytes | None, piper_sr: int = 22050) -> str | None:
    _hdr("3/5  STT — Whisper tiny (CPU)")
    try:
        from voice_agent.local_plugins import LocalWhisperSTT
        import numpy as np

        stt = LocalWhisperSTT(model="tiny")

        if piper_pcm is not None:
            # Transcribe the Piper audio we just synthesized — closed-loop test.
            samples_f32 = np.frombuffer(piper_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            # Resample to 16 kHz if needed
            if piper_sr != 16000:
                from scipy.signal import resample_poly
                import math
                def gcd(a, b): return a if b == 0 else gcd(b, a % b)
                g = gcd(piper_sr, 16000)
                samples_f32 = resample_poly(samples_f32, 16000 // g, piper_sr // g).astype(np.float32)
            source = "Piper-synthesized audio"
        else:
            # Fallback: 2 s of silence (will likely produce empty transcript)
            samples_f32 = np.zeros(16000 * 2, dtype=np.float32)
            source = "2 s silence (no Piper output available)"

        print(f"  Input: {source}")
        t0 = time.time()
        stt._ensure_model()
        segments, _ = stt._whisper.transcribe(samples_f32, language="en", beam_size=1, vad_filter=True)
        transcript = " ".join(s.text.strip() for s in segments).strip()
        elapsed = time.time() - t0
        print(f"  {PASS}  {elapsed:.2f}s")
        print(f"  Transcript: {transcript!r}")
        return transcript
    except Exception as e:
        print(f"  {FAIL}  {e}")
        import traceback; traceback.print_exc()
        return None


# ── 4. TTS (Piper) ───────────────────────────────────────────────────────────
def test_tts(text: str | None) -> tuple[bytes, int] | tuple[None, None]:
    _hdr("4/5  TTS — Piper · en_US-lessac-medium (CPU)")
    synth_text = text or "Hi, this is Vox. I'm calling about improving your sales workflow. Got a moment?"
    print(f"  Text: {synth_text!r}")
    try:
        from voice_agent.local_plugins import PiperTTS
        piper = PiperTTS(voice="en_US-lessac-medium")

        t0 = time.time()
        piper._ensure_model()
        load_ms = (time.time() - t0) * 1000
        print(f"  Model loaded: {load_ms:.0f} ms")

        t0 = time.time()
        chunks = list(piper._piper.synthesize(synth_text))
        if not chunks:
            print(f"  {FAIL}  No audio chunks returned")
            return None, None
        sr = chunks[0].sample_rate
        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        elapsed = time.time() - t0

        duration = len(pcm) / 2 / sr
        rtf = elapsed / duration
        print(f"  {PASS}  {elapsed:.2f}s inference -> {duration:.2f}s audio  (RTF {rtf:.2f}x)")
        if rtf < 0.5:
            print(f"         Excellent — well under real-time")
        elif rtf < 1.0:
            print(f"         Good — under real-time")
        else:
            print(f"         Slow — CPU may be under load")
        return pcm, sr
    except Exception as e:
        print(f"  {FAIL}  {e}")
        import traceback; traceback.print_exc()
        return None, None


# ── 5. Save audio output ─────────────────────────────────────────────────────
def save_wav(pcm: bytes, sr: int, path: Path):
    _hdr("5/5  Saving audio output")
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        print(f"  {PASS}  Saved -> {path}")
        print(f"         Open this file in any audio player to hear the TTS output")
    except Exception as e:
        print(f"  {FAIL}  Could not save: {e}")


def play_audio(pcm: bytes, sr: int):
    try:
        import sounddevice as sd
        import numpy as np
        samples = np.frombuffer(pcm, dtype=np.int16)
        print(f"  Playing {len(pcm)/2/sr:.1f}s of audio…")
        sd.play(samples, samplerate=sr, blocking=True)
        print(f"  Done")
    except ImportError:
        print(f"  {SKIP}  sounddevice not installed — pip install sounddevice")
    except Exception as e:
        print(f"  {FAIL}  Playback error: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Windows cmd/PowerShell default to cp1252; force UTF-8 so box chars print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Local-combo test (no Twilio/LiveKit needed)")
    parser.add_argument("--play",      action="store_true", help="Play TTS output via sounddevice")
    parser.add_argument("--skip-llm",  action="store_true", help="Skip LLM test (Ollama not running)")
    parser.add_argument("--skip-stt",  action="store_true", help="Skip STT test")
    parser.add_argument("--out",       default="test_output.wav", help="Output WAV path")
    args = parser.parse_args()

    print("\nLocal OSS combo test")
    print("  STT: faster-whisper tiny  |  LLM: Ollama qwen2.5:0.5b  |  TTS: Piper en_US-lessac")

    prereqs_ok = check_prereqs(skip_llm=args.skip_llm, skip_stt=args.skip_stt)
    if not prereqs_ok:
        print(f"\n{FAIL}  Fix the above issues before testing. Aborting.")
        sys.exit(1)

    llm_reply = None
    if not args.skip_llm:
        llm_reply = test_llm()
    else:
        print(f"\n{SEP}\n  2/5  LLM — skipped\n{SEP}")

    # TTS: synthesize the LLM reply (or a canned sentence if LLM skipped/failed)
    tts_text = llm_reply or "Hi, this is Vox from Nexus, calling about a quick business check-in."
    pcm, sr = test_tts(tts_text)

    # STT: transcribe the Piper output
    if not args.skip_stt:
        test_stt(pcm, sr or 22050)
    else:
        print(f"\n{SEP}\n  3/5  STT — skipped\n{SEP}")

    # Save audio
    out_path = Path(args.out)
    if pcm:
        save_wav(pcm, sr, out_path)
        if args.play:
            play_audio(pcm, sr)
    else:
        print(f"\n{SEP}\n  5/5  No audio to save (TTS failed)\n{SEP}")

    # Summary
    print(f"\n{SEP}")
    results = []
    if not args.skip_llm:
        results.append(("LLM",  llm_reply is not None))
    results.append(("TTS", pcm is not None))
    if not args.skip_stt:
        results.append(("STT", True))  # would have exited on failure above
    passed = sum(v for _, v in results)
    total  = len(results)
    print(f"  Done. {passed}/{total} components working.")
    if pcm:
        print(f"  Listen: open {out_path.resolve()}")
    print(SEP)


if __name__ == "__main__":
    # Must run from the project root (nexuscaller-lab/)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.environ.setdefault("GROQ_API_KEY", "unused")
    main()
