"""
Outbound voice-agent module — Phase 1 POC.

Bridges a Twilio outbound call to a Pipecat pipeline (Groq STT + Llama
LLM + ElevenLabs TTS), records the transcript, then summarises the call
via Groq Llama once the caller hangs up.

Layout:
    pipeline.py    Pipecat task built around TwilioFrameSerializer
    server.py      FastAPI: TwiML + Media Streams WebSocket
    dial.py        CLI that initiates the outbound call
    summary.py     Post-call structured summary
    transcripts/   JSONL per call (created on first run)
"""
