"""
NexusCaller voice-agent module (LiveKit edition).

Outbound PSTN calling via LiveKit SIP trunks + cloud or local STT/LLM/TTS.
Each call is dispatched as a LiveKit agent job to this worker.

Layout:
    agent.py          LiveKit agent worker — STT/LLM/TTS pipeline per call
    server.py         FastAPI: precall UI, /api/dial, cockpit, /api/catalog
    combos.py         Preset stack combos shown in the precall picker
    local_plugins.py  Offline STT (Whisper) and TTS (Piper) wrappers
    storage.py        Transcript + summary file paths
    summary.py        Post-call structured summary via Groq Llama
    transcripts/      JSONL per call (created on first run)
"""
