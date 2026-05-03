# Vox — Voice Agent (LiveKit edition)

```
NexusAgent CRM       nexuscaller-lab
──────────────       ────────────────
[Call] button  ─────► /precall page (combo picker)
                            │
                            ▼
                     POST /api/dial
                            │
                            ▼
                    LiveKit Cloud (rooms + Krisp AEC)
                            │      ▲
       agent worker ────────┘      └────── cockpit /calls/{id} (subscribes for live transcript)
       (Vox brain)
                            │
                            ▼
                       SIP outbound trunk
                            │
                            ▼
                    Twilio Elastic SIP Trunk → PSTN → caller's phone
```

The audio bridge is now LiveKit, not Pipecat over Twilio Media Streams.
LiveKit gives us **Krisp echo cancellation** + sophisticated turn detection,
which solves the self-interruption + barge-in problems we hit with the old
stack.

## Files

| File | What it does |
|---|---|
| `agent.py` | LiveKit Agent worker — runs the conversation. Picks up jobs dispatched by the server, builds STT/LLM/TTS plugins from the per-call combo, runs `AgentSession`. |
| `server.py` | FastAPI: `/precall` HTML, `/api/dial` (creates room + dispatches agent + initiates SIP call), `/calls/{id}` cockpit, `/api/cockpit-token/{id}` issues a viewer JWT. |
| `combos.py` | Preset stack combos (Groq+ElevenLabs, Deepgram+ElevenLabs, Deepgram+Cartesia) + dropdown options for manual override. |
| `summary.py` | Post-call Groq Llama summarizer (lead-gen JSON: outcome, lead_score, etc.). Unchanged. |
| `storage.py` | Path helpers for transcript JSONL files. Unchanged. |
| `precall.html` | Operator-facing combo picker. JS reads call payload from URL hash. |
| `cockpit.html` | Live transcript view. Connects to LiveKit room as a viewer participant via the JS SDK; receives turns from the agent's data channel. |

## One-time setup (your action)

You'll do this **once**, then `start_all.bat` boots everything.

### 1. Sign up at LiveKit Cloud (free)

- Go to https://cloud.livekit.io
- Sign up (Google or email)
- Create a project (any name, e.g. `vox-dev`)
- Note your **WebSocket URL**: `wss://<project-id>.livekit.cloud`
- Project Settings → **Keys** → Generate API key + secret
  - **API Key** (starts with `API…`)
  - **API Secret** (long random string)

Put these in `.env`:

```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxxxx
LIVEKIT_API_SECRET=yyyyyyyyyyyyyyyyy
```

### 2. Create an outbound SIP trunk in LiveKit

The agent calls phones via **LiveKit's SIP service** which forwards to Twilio.

- LiveKit Cloud → Project → **Telephony** → **SIP Trunks**
- Click **Create outbound trunk**
- Use these settings:
  - **Name**: `twilio-out`
  - **Address**: `sip:<your-twilio-trunk>.pstn.twilio.com` (you'll get this in step 3)
  - **Numbers**: any list including your Twilio number `+16184256045`
  - **Authentication**: leave as IP allowlist (Twilio will whitelist LiveKit's IPs)
- After creating: copy the **Trunk ID** (looks like `ST_xxxxxxxxxxxx`).

Put it in `.env`:

```
LIVEKIT_OUTBOUND_TRUNK_ID=ST_xxxxxxxxxxxx
```

### 3. Create a Twilio Elastic SIP Trunk

Twilio side — receives the SIP INVITE from LiveKit and forwards to PSTN.

- Twilio Console → **Voice** → **Elastic SIP Trunking** → **Trunks**
- Click **Create new SIP trunk** (named e.g. `livekit-bridge`, ~$1/month)
- In the trunk:
  - **Termination** tab → set a Termination URI (e.g. `your-trunk.pstn.twilio.com`).
    Copy this — paste into LiveKit's trunk **Address** field (step 2).
  - **Authentication** tab → add LiveKit Cloud's egress IPs
    (LiveKit shows them when you create the trunk; current list: `44.237.62.249/32` etc.)
  - **Origination** tab → leave empty (we're outbound only)
  - **Numbers** tab → click **Associate phone number** and add your Twilio
    `+16184256045` so the From: header matches a verified Twilio number.

### 4. Verify

```powershell
cd "C:\Users\Praneeth p\OneDrive\Desktop\nexuscaller-lab"
.\venv\Scripts\Activate.ps1
python -m voice_agent.agent dev
```

You should see `registered worker` in the output — that means the agent
successfully connected to LiveKit Cloud. If it errors out, double-check
`LIVEKIT_URL/KEY/SECRET` in `.env`.

## Daily use

Just double-click **`start_all.bat`** in the lab folder.
Five windows open: agent worker, voice-agent server, NexusAgent API,
NexusAgent frontend, and ngrok. When you're done, press any key in the
launcher window to close them all.

## To make a call

1. Open `http://localhost:5173` (NexusAgent UI)
2. Sign in
3. CRM → Contacts → pick a contact (must have a phone in E.164 format like `+91...`)
4. Click **"Call with Vox"**
5. Modal asks for purpose → click **Place call**
6. New tab opens at `/precall` → pick a combo → click **Place call**
7. Browser redirects to `/calls/<id>` cockpit → live transcript fills in
8. Phone rings, talk to Vox, hang up
9. Cockpit shows the call summary; back in NexusAgent the contact's
   "Vox calls" panel updates with the new entry

## What's gone (Pipecat era)

We removed:
- `voice_agent/pipeline.py` (Pipecat task builder + TwilioFrameSerializer)
- `voice_agent/transcript.py` (Pipecat-frame transcript collector)
- `voice_agent/stt_mute.py` (half-duplex echo workaround — Krisp does it now)
- `voice_agent/dial.py` (Twilio-direct dial CLI)
- `POST /twilio/voice` and `WS /twilio/stream` endpoints
- `--reload`-incompatible audio buffering hacks

LiveKit handles all the audio so we don't need any of that anymore.

## Troubleshooting

**Agent worker exits with `connection refused`** → `LIVEKIT_URL` wrong or
project doesn't exist yet. Verify in LiveKit Cloud dashboard.

**`/api/dial` returns 502 "LiveKit dispatch failed"** → API key/secret
mismatch. Regenerate keys in LiveKit Cloud and update `.env`.

**Phone rings but no audio** → SIP trunk not configured correctly. Check:
(a) LiveKit outbound trunk address matches Twilio termination URI,
(b) LiveKit's egress IPs are whitelisted in Twilio's authentication tab,
(c) Twilio number associated with the trunk matches `TWILIO_PHONE_NUMBER`.

**Phone doesn't ring at all** → Twilio trial-account verified-number
list still applies. Make sure the destination is on
Twilio Console → Phone Numbers → Verified Caller IDs.
