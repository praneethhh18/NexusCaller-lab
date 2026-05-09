"""
Quick diagnostic: verify the agent worker config, API keys, and stack.
Run: .\venv\Scripts\python -m voice_agent.diagnose
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check_groq(key: str) -> bool:
    payload = json.dumps({
        "model": "llama-3.1-8b-instant",
        "stream": False,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = json.loads(r.read())
        reply = body["choices"][0]["message"]["content"].strip()
        print(f"  {PASS}  Groq LLM OK  reply={reply!r}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"  {FAIL}  Groq 401 Unauthorized (invalid key)")
        elif e.code == 403:
            print(f"  {FAIL}  Groq 403 Forbidden  --- KEY EXPIRED / REVOKED ---")
            print("         Go to https://console.groq.com/keys -- create a new key")
            print("         Then update GROQ_API_KEY in .env and restart the worker window")
        elif e.code == 429:
            print(f"  {PASS}  Groq --429 rate-limited (key is valid, just throttled)")
            return True
        else:
            print(f"  {FAIL}  Groq --HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"  {FAIL}  Groq --{e}")
        return False


def check_elevenlabs(key: str) -> bool:
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/user",
        headers={"xi-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
        tier = body.get("subscription", {}).get("tier", "?")
        chars = body.get("subscription", {}).get("character_count", "?")
        limit = body.get("subscription", {}).get("character_limit", "?")
        print(f"  {PASS}  ElevenLabs OK  tier={tier!r}  chars_used={chars}/{limit}")
        return True
    except urllib.error.HTTPError as e:
        print(f"  {FAIL}  ElevenLabs --HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"  {FAIL}  ElevenLabs --{e}")
        return False


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("\n=== NexusCaller Agent Diagnostics ===\n")

    lk_url    = os.getenv("LIVEKIT_URL", "")
    lk_key    = os.getenv("LIVEKIT_API_KEY", "")
    lk_secret = os.getenv("LIVEKIT_API_SECRET", "")
    lk_agent  = os.getenv("LIVEKIT_AGENT_NAME", "vox")
    groq_key  = os.getenv("GROQ_API_KEY", "")
    el_key    = os.getenv("ELEVENLABS_API_KEY", "")

    # 1. Env vars
    print("[1] Environment")
    print(f"  LIVEKIT_URL         = {lk_url or '*** MISSING ***'}")
    ok_lk = bool(lk_url and lk_key and lk_secret)
    print(f"  LIVEKIT_API_KEY     = {'SET (' + lk_key[:8] + '...)' if lk_key else '*** MISSING ***'}")
    print(f"  LIVEKIT_API_SECRET  = {'SET' if lk_secret else '*** MISSING ***'}")
    print(f"  LIVEKIT_AGENT_NAME  = {lk_agent!r}  (must match CreateAgentDispatchRequest)")
    print(f"  GROQ_API_KEY        = {'SET (' + groq_key[:8] + '...)' if groq_key else '*** MISSING ***'}")
    print(f"  ELEVENLABS_API_KEY  = {'SET' if el_key else '*** MISSING ***'}")

    # 2. Plugin imports
    print("\n[2] Plugin imports")
    try:
        from livekit.agents import WorkerOptions
        # Probe each plugin loads — diagnostic only, not used here.
        from livekit.plugins import cartesia, deepgram, elevenlabs, openai, silero  # noqa: F401
        print(f"  {PASS}  livekit-agents + all plugins")
    except ImportError as e:
        print(f"  {FAIL}  {e}")
        sys.exit(1)

    # 3. WorkerOptions agent_name
    print("\n[3] WorkerOptions")
    from voice_agent.agent import entrypoint, prewarm
    opts = WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name=lk_agent)
    print(f"  agent_name = {opts.agent_name!r}  {'OK — matches LIVEKIT_AGENT_NAME' if opts.agent_name == lk_agent else 'MISMATCH!'}")

    # 4. API keys
    print("\n[4] API keys (live check)")
    groq_ok = False
    el_ok = False
    if groq_key:
        groq_ok = check_groq(groq_key)
    else:
        print(f"  {SKIP}  Groq (no key)")
    if el_key:
        el_ok = check_elevenlabs(el_key)
    else:
        print(f"  {SKIP}  ElevenLabs (no key)")

    # 5. Default combo
    print("\n[5] Default combo (groq-elevenlabs)")
    from voice_agent.combos import default_combo
    c = default_combo()
    print(f"  key={c.key!r}  stt={c.stt!r}  llm={c.llm!r}  tts={c.tts!r}")

    # 6. Summary
    print("\n=== Summary ===")
    issues = []
    if not ok_lk:
        issues.append("LiveKit credentials missing in .env")
    if not groq_ok:
        issues.append("Groq API key invalid — renew at https://console.groq.com/keys")
    if not el_ok:
        issues.append("ElevenLabs API key invalid — check at https://elevenlabs.io")
    if issues:
        for i in issues:
            print(f"  {FAIL}  {i}")
        print()
        print("Fix the above, update .env, then:")
    else:
        print(f"  {PASS}  All checks passed!")
        print()
        print("To apply changes:")
    print("  1. Close the 'Lab · Vox Agent worker' window")
    print("  2. Re-run start_all.bat  (or just reopen that window)")
    print("  3. Wait for: [info] worker registered, agent_name='vox'")
    print("  4. Place a test call via http://localhost:8765/precall")
    print()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
