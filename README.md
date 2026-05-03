# NexusCaller-lab

Outbound voice agent that **NexusAgent**'s CRM dials through. The CRM owns
contacts and history; this repo owns the audio path — Twilio for the PSTN
leg, LiveKit for the audio bridge + echo cancellation, and a stack of
swappable cloud STT / LLM / TTS plugins for the conversation itself.

> Two projects, one connected system. NexusAgent ↔ this lab talk over HTTP.
> The CRM stays lean (no audio deps); the lab stays focused on telephony.

```
NexusAgent CRM (other repo)             nexuscaller-lab (this repo)
──────────────────────────              ────────────────────────────
  Contact: Asha Singh
    [☎ Call with Vox] ────POST────►  /precall — operator picks combo
                                            │
                                            ▼
                                       /api/dial
                                            │
                                            ▼
                                LiveKit Cloud  ◄── cockpit /calls/<id>
                                  (rooms +          (live transcript)
                                  Krisp AEC)
                                            │
                          agent worker ─────┘
                          (Vox brain — STT/LLM/TTS)
                                            │
                                            ▼
                                  Twilio SIP trunk
                                            │
                                            ▼
                                          PSTN
                                            │
                                            ▼
                                  caller's phone

  ◄──────── webhook ── transcript + summary
  Contact history shows the new call (auto)
```

## What's in this repo

```
voice_agent/
├── agent.py        # LiveKit Agent worker — the conversation loop
├── server.py       # FastAPI: /precall, /api/dial, /calls/<id>, /api/cockpit-token
├── combos.py       # 3 preset stacks + dropdown options for manual override
├── summary.py      # Post-call lead-gen summarizer (Groq Llama → JSON)
├── storage.py      # Per-call JSONL + summary file paths
├── precall.html    # Operator's pre-dial config page
├── cockpit.html    # Live transcript viewer (LiveKit JS subscribe-only)
└── README.md       # Detailed setup walkthrough — read this for first-time setup
start_all.bat       # One-click launcher (5 windows: agent + lab + NexusAgent + frontend + ngrok)
stop_all.bat        # Kill everything
.env.example        # All env vars, commented
```

That's it. The browser-mic Pipecat playground that earlier versions of this
project were is gone — see [git history](https://github.com/praneethhh18/NexusCaller-lab/commits/main)
if you need it. The current product is the LiveKit-based dialer above.

## The three preset combos

Picked from the precall page before each call. Manual override via the
dropdowns inside the "Customize stack" disclosure if you want a non-preset
combination.

| Combo | STT | LLM | TTS | When to use |
|---|---|---|---|---|
| **Groq + ElevenLabs** ★ default | Groq Whisper Large-v3-turbo | Groq Llama 3.1 8B Instant | ElevenLabs Turbo v2.5 | Free-tier baseline. No monthly cap on Groq. |
| **Deepgram + ElevenLabs** | Deepgram Nova-3 | Groq Llama 3.1 8B Instant | ElevenLabs Turbo v2.5 | Best Indian-English transcription. Recommended for Indian callers. |
| **Deepgram + Cartesia** | Deepgram Nova-3 | Groq Llama 3.1 8B Instant | Cartesia Sonic-3 | Most natural-sounding voice. |

Add or modify combos in `voice_agent/combos.py`.

## Quick start

If you've never set the project up before, follow
[`voice_agent/README.md`](./voice_agent/README.md) — it walks you through
LiveKit Cloud + Twilio SIP trunk creation in detail.

Once your `.env` has the LiveKit + Twilio + AI-provider keys, daily use is:

```bat
:: From the project root
start_all.bat
```

Five terminal windows pop up:
1. **Lab · Vox Agent worker** — registers with LiveKit and waits for jobs
2. **Lab · Voice agent (8765)** — FastAPI for precall + cockpit
3. **NexusAgent · API (8000)** — the CRM backend (in the sibling repo)
4. **NexusAgent · Frontend (5173)** — the React UI
5. **ngrok · tunnel** — public HTTPS for the lab (only needed if you want to share the cockpit URL)

When you're done, press any key in the launcher window — it closes all four
spawned windows. `stop_all.bat` is the manual fallback if anything got stuck.

## How a call works

1. Open NexusAgent at `http://localhost:5173`, log in
2. **CRM → Contacts → click a contact** (must have a phone number in E.164
   format, e.g. `+91...`)
3. Click **"Call with Vox"** → the modal asks for the call's purpose
4. Click **Place call** → a new tab opens at `http://localhost:8765/precall?...`
5. **Pick a combo** (or customize via the dropdown) → click **Place call**
6. Browser redirects to `http://localhost:8765/calls/<call_id>` — the live
   cockpit. Orb is connecting.
7. Phone rings → caller answers → Vox speaks the greeting
8. Conversation flows. The cockpit shows transcript turns live.
9. Caller hangs up → cockpit shows "Call ended" + the summary card
   (outcome, lead score, key points, next step) within ~5 seconds
10. Back in NexusAgent, the contact's **Vox calls** panel updates with the
    new call — full transcript + structured summary stored in the CRM DB
    via the callback.

## Required env vars

The full list is in `.env.example`. Bare minimum:

```bash
# AI providers
GROQ_API_KEY=gsk_...
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL   # Sarah (premade voice)

# Optional (for the Deepgram + Cartesia combo)
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
CARTESIA_VOICE_ID=...

# Twilio — outbound PSTN
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...

# LiveKit — audio bridge + Krisp echo cancellation
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=API...
LIVEKIT_API_SECRET=...
LIVEKIT_OUTBOUND_TRUNK_ID=ST_...

# Voice-agent server
VOICE_AGENT_PORT=8765
PUBLIC_URL=                               # ngrok URL, optional
VOICE_CALLBACK_SECRET=                    # shared with NexusAgent
```

## What's intentionally NOT here

- **No CRM, no contacts, no business logic.** Those live in NexusAgent.
- **No hosting infrastructure.** This runs on your laptop during dev. Real
  deployment is a future concern — LiveKit Cloud handles the heavy
  multi-tenant audio side.
- **No browser-mic playground.** Earlier revisions had a Pipecat-based UI
  for testing model combos with your laptop mic. Removed when we migrated
  to LiveKit. If you want a model A/B test rig, build one separately or
  open the lab cockpit on a finished call to inspect transcripts.

## Cost notes

- **LiveKit Cloud:** free tier covers ~1,000 SIP-minutes / month + generous
  bandwidth. Plenty for development and small-team production.
- **Twilio:** pay per outbound minute (~$0.013/min US, ~$0.20/min India).
  $1/month for the SIP trunk rental on top.
- **Groq:** free tier, no monthly cap as of writing.
- **ElevenLabs:** free tier = 10,000 characters/month (~10 minutes of
  speech). Hits the cap fast on real-world traffic — upgrade or switch
  to Cartesia.
- **Deepgram:** $200 free credit at signup ≈ 776 hours of streaming
  transcription. Hard to burn through.
- **Cartesia:** free trial credits at signup.

The PSTN call itself (Twilio) is the only piece you genuinely cannot get
for free — every carrier charges. Everything else covers free-tier dev work.

## License

MIT — see the parent repo for the full license text.

## Companion repository

This repo is one half of a pair. The CRM half — Vox's caller, NexusAgent —
lives at https://github.com/praneethhh18/Nexus.
