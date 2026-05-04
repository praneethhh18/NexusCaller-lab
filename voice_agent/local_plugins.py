"""
Local-only LiveKit plugins for the OSS combo.

LiveKit doesn't ship plugins for `faster-whisper` or `kokoro-onnx`, so we
wrap them ourselves behind the standard `livekit.agents.stt.STT` and
`livekit.agents.tts.TTS` abstract classes. That way the agent worker
doesn't care whether the STT/TTS came from Groq's cloud, Deepgram, or
this fully-offline path — same plugin contract.

Trade-offs vs the cloud combos:
  • Whisper-tiny on CPU: ~1-2× real-time. So a 3 s utterance takes
    ~3-6 s to transcribe. Noticeable on a phone call.
  • Kokoro on CPU: ~0.5-1× real-time generation, plus ONNX startup cost.
  • Quality: Whisper-tiny is markedly worse than Whisper-large at
    Indian-English / acronyms; Kokoro voices are decent but less
    expressive than ElevenLabs.
  • Offline-ness: zero cloud calls for STT / TTS once models are
    downloaded. The LLM path uses Ollama which is already local.

Use this combo for cost experiments, demos without internet, or as a
hard fallback when cloud providers are degraded.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
from livekit.agents import (
    APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr,
    stt, tts, utils,
)
from loguru import logger


# ── Shared model cache directory ─────────────────────────────────────────
_CACHE_DIR = Path(os.getenv(
    "LOCAL_MODEL_CACHE",
    Path.home() / ".cache" / "nexuscaller-local-models",
))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Local Whisper STT (faster-whisper)
# ═══════════════════════════════════════════════════════════════════════════
class LocalWhisperSTT(stt.STT):
    """
    faster-whisper running on CPU. The model auto-downloads from HF on
    first use into the system's HF cache (~/.cache/huggingface/hub).

    We deliberately use `compute_type="int8"` to halve memory + speed up
    inference at a small accuracy cost — fine for tiny.
    """

    def __init__(
        self,
        *,
        model: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._model_name = model
        self._language = language
        self._whisper = None  # lazy-init on first recognize
        self._device = device
        self._compute_type = compute_type
        logger.info(f"[local-whisper] configured · model={model} device={device} compute={compute_type}")

    def _ensure_model(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel
            logger.info(f"[local-whisper] loading WhisperModel({self._model_name!r}) — first call may download weights")
            self._whisper = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=int(os.getenv("LOCAL_WHISPER_THREADS", "4")),
            )

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        # Combine the AudioBuffer's frames into one contiguous PCM array.
        combined = utils.audio.combine_audio_frames(buffer)
        # combined.data is a memoryview of int16 PCM at combined.sample_rate Hz.
        pcm16 = np.frombuffer(combined.data, dtype=np.int16)
        # faster-whisper wants float32 in [-1, 1] at 16 kHz mono.
        samples = pcm16.astype(np.float32) / 32768.0
        # Resample if the input rate isn't 16 kHz (Twilio is 8 kHz μ-law).
        if combined.sample_rate != 16000:
            try:
                from scipy.signal import resample_poly
                gcd = np.gcd(int(combined.sample_rate), 16000)
                up, down = 16000 // gcd, int(combined.sample_rate) // gcd
                samples = resample_poly(samples, up, down).astype(np.float32)
            except ImportError:
                # Fall back to numpy stride resample (fine for integer-ratio rates).
                ratio = 16000 / combined.sample_rate
                idx = (np.arange(int(len(samples) * ratio)) / ratio).astype(np.int32)
                samples = samples[np.clip(idx, 0, len(samples) - 1)]

        # Run inference off the event loop — faster-whisper is sync + heavy.
        text = await asyncio.to_thread(self._transcribe, samples, language)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=self._language)],
        )

    def _transcribe(self, samples: np.ndarray, language) -> str:
        self._ensure_model()
        lang = language if isinstance(language, str) else self._language
        segments, _info = self._whisper.transcribe(
            samples,
            language=lang,
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Local Kokoro TTS (kokoro-onnx, CPU)
# ═══════════════════════════════════════════════════════════════════════════
# Default model + voices files. kokoro-onnx ships no auto-download, so we
# fetch from the project's HuggingFace mirror on first use. ~150 MB total.
_KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
_KOKORO_MODEL_PATH = _CACHE_DIR / "kokoro-v1.0.onnx"
_KOKORO_VOICES_PATH = _CACHE_DIR / "voices-v1.0.bin"


def _ensure_kokoro_files() -> None:
    """Download Kokoro model + voice files on first use. One-time, ~150 MB."""
    import urllib.request

    for url, dest in [(_KOKORO_MODEL_URL, _KOKORO_MODEL_PATH),
                      (_KOKORO_VOICES_URL, _KOKORO_VOICES_PATH)]:
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            continue
        logger.info(f"[kokoro] downloading {url} → {dest} (one-time)")
        urllib.request.urlretrieve(url, dest)


class KokoroTTS(tts.TTS):
    """
    Kokoro-82M via ONNX Runtime. CPU-only. Pretty natural for an OSS
    voice. ~150 MB of model files cached under ~/.cache/nexuscaller-local-models/
    on first use.

    Output sample rate is 24 kHz mono PCM — LiveKit's transport handles
    resampling to whatever the SIP leg negotiates (8 kHz μ-law for Twilio).
    """

    DEFAULT_VOICE = "af_bella"   # warm female English voice

    def __init__(self, *, voice: str = DEFAULT_VOICE, sample_rate: int = 24000):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._voice = voice
        self._kokoro = None  # lazy-init
        logger.info(f"[kokoro] configured · voice={voice} rate={sample_rate}")

    def _ensure_model(self):
        if self._kokoro is None:
            _ensure_kokoro_files()
            from kokoro_onnx import Kokoro
            logger.info(f"[kokoro] loading ONNX session from {_KOKORO_MODEL_PATH}")
            self._kokoro = Kokoro(str(_KOKORO_MODEL_PATH), str(_KOKORO_VOICES_PATH))

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _KokoroChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _KokoroChunkedStream(tts.ChunkedStream):
    """One-shot synthesis: compute the whole utterance, push as one segment."""

    async def _run(self, output_emitter):
        kokoro_tts: KokoroTTS = self._tts  # type: ignore[assignment]
        text = self._input_text

        # Off the event loop — onnxruntime inference is sync + CPU-heavy.
        samples, sr = await asyncio.to_thread(self._synthesize_blocking, kokoro_tts, text)
        if samples is None:
            return

        # Convert float32 samples in [-1, 1] → int16 PCM bytes.
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=sr,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())
        output_emitter.push(pcm16)
        output_emitter.end_segment()
        output_emitter.flush()

    @staticmethod
    def _synthesize_blocking(kokoro_tts: KokoroTTS, text: str):
        kokoro_tts._ensure_model()
        try:
            samples, sr = kokoro_tts._kokoro.create(text, voice=kokoro_tts._voice)
            return samples, sr
        except Exception as e:
            logger.exception(f"[kokoro] synthesis failed: {e}")
            return None, kokoro_tts.sample_rate


# ═══════════════════════════════════════════════════════════════════════════
#  Local Piper TTS (piper-tts, CPU, real-time capable)
# ═══════════════════════════════════════════════════════════════════════════
# Piper voices are ~30-60 MB each (ONNX + JSON config). Much lighter than
# Kokoro's 150 MB, and typically faster than real-time on a modern CPU.
# Voices are cached under ~/.cache/nexuscaller-local-models/piper/.

_PIPER_DIR = _CACHE_DIR / "piper"
_PIPER_DIR.mkdir(parents=True, exist_ok=True)

# Catalog of known voices: voice_key → (onnx_url, config_url, sample_rate)
# Full list at https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0
_PIPER_VOICE_CATALOG: dict[str, tuple[str, str, int]] = {
    "en_US-lessac-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        22050,
    ),
    "en_US-ryan-high": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
        22050,
    ),
    "en_US-arctic-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/arctic/medium/en_US-arctic-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json",
        22050,
    ),
    "en_GB-alan-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        22050,
    ),
}
_PIPER_DEFAULT_VOICE = "en_US-lessac-medium"


def _ensure_piper_voice(voice: str) -> tuple[Path, Path]:
    """Download the ONNX model + JSON config for a Piper voice. ~30-60 MB."""
    import urllib.request

    if voice not in _PIPER_VOICE_CATALOG:
        raise ValueError(
            f"Unknown Piper voice {voice!r}. "
            f"Known voices: {list(_PIPER_VOICE_CATALOG)}"
        )
    onnx_url, config_url, _ = _PIPER_VOICE_CATALOG[voice]
    onnx_path = _PIPER_DIR / f"{voice}.onnx"
    config_path = _PIPER_DIR / f"{voice}.onnx.json"
    for url, dest in [(onnx_url, onnx_path), (config_url, config_path)]:
        if dest.exists() and dest.stat().st_size > 1024:
            continue
        logger.info(f"[piper] downloading {dest.name} (one-time)…")
        urllib.request.urlretrieve(url, dest)
    return onnx_path, config_path


class PiperTTS(tts.TTS):
    """
    Piper-TTS via ONNX Runtime. CPU-only, typically faster than real-time.
    Voices are ~30-60 MB, downloaded on first use to
    ~/.cache/nexuscaller-local-models/piper/.

    Default voice: en_US-lessac-medium (clear, neutral American English).
    Other options: en_US-ryan-high (male), en_GB-alan-medium (British).
    """

    def __init__(self, *, voice: str = _PIPER_DEFAULT_VOICE):
        sr = _PIPER_VOICE_CATALOG.get(voice, ("", "", 22050))[2]
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sr,
            num_channels=1,
        )
        self._voice_name = voice
        self._piper = None  # lazy-init
        logger.info(f"[piper] configured · voice={voice} rate={sr}")

    def _ensure_model(self):
        if self._piper is None:
            onnx_path, config_path = _ensure_piper_voice(self._voice_name)
            from piper.voice import PiperVoice
            logger.info(f"[piper] loading ONNX session from {onnx_path.name}")
            self._piper = PiperVoice.load(
                str(onnx_path),
                config_path=str(config_path),
                use_cuda=False,
            )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _PiperChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _PiperChunkedStream(tts.ChunkedStream):
    """One-shot synthesis: run Piper ONNX inference, push raw PCM as one segment."""

    async def _run(self, output_emitter):
        piper_tts: PiperTTS = self._tts  # type: ignore[assignment]
        text = self._input_text

        pcm_bytes, sr = await asyncio.to_thread(
            self._synthesize_blocking, piper_tts, text
        )
        if pcm_bytes is None:
            return

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=sr,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())
        output_emitter.push(pcm_bytes)
        output_emitter.end_segment()
        output_emitter.flush()

    @staticmethod
    def _synthesize_blocking(piper_tts: PiperTTS, text: str):
        piper_tts._ensure_model()
        try:
            # piper-tts ≥ 2.0 returns Iterable[AudioChunk]; concatenate PCM bytes.
            chunks = list(piper_tts._piper.synthesize(text))
            if not chunks:
                logger.warning("[piper] synthesis returned no audio chunks")
                return None, piper_tts.sample_rate
            sr = chunks[0].sample_rate
            pcm_bytes = b"".join(c.audio_int16_bytes for c in chunks)
            return pcm_bytes, sr
        except Exception as e:
            logger.exception(f"[piper] synthesis failed: {e}")
            return None, piper_tts.sample_rate
