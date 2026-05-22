"""Language pack for Vox.

Hindi / Tamil / Marathi / English in one place so the agent code stays
flat — each language is a dict of (greeting_variants, system_prompt
fragments, filler_words). The agent reads `meta["language"]` (default
"en") and pulls from here.

Voice IDs are Cartesia's multilingual catalog (sonic-multilingual model
required). Add IDs here when we add a language; do not scatter them
through the code.

To add a language:
  1. Add an entry to LANGUAGES
  2. Translate the greetings + the three prompt blocks
  3. Pick a Cartesia voice from
     https://docs.cartesia.ai/api-reference/voices/list
     that lists your language in `language` metadata
  4. (optional) Override BACKCHANNELS so "mmhmm" sounds native
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# Cartesia model that supports all the languages we ship.
CARTESIA_MULTILINGUAL_MODEL = "sonic-multilingual"


@dataclass
class LanguageProfile:
    """Per-language voice + copy pack."""
    code: str                              # ISO 639-1 ("en", "hi", "ta", "mr")
    display_name: str                      # "English", "हिन्दी", "தமிழ்", "मराठी"
    cartesia_voice_id: Optional[str]       # native-sounding voice on Cartesia
    elevenlabs_voice_id: Optional[str]     # fallback if Cartesia not configured
    # Three short greeting variants per relationship type so back-to-back
    # demo calls don't sound canned. {name}, {biz}, {purpose}, {agent}
    # are filled in by _greeting at call time.
    greetings_customer: List[str]          # warm, returning-customer tone
    greetings_known_lead: List[str]        # we have their name, no prior chat
    greetings_cold: List[str]              # no name, no context
    # Prepended to the system prompt so the LLM knows what language to
    # reply in and how formal to be for that language's culture.
    speak_instruction: str
    busy_handler_block: str                # the "if they say make-it-fast" block
    backchannels: List[str] = field(default_factory=list)  # mmhmm, gotcha etc.


LANGUAGES = {
    # ──────────────────────────────────────────────────────────────────────
    "en": LanguageProfile(
        code="en",
        display_name="English",
        cartesia_voice_id=None,            # use CARTESIA_VOICE_ID env default
        elevenlabs_voice_id=None,          # use ELEVENLABS_VOICE_ID env default
        greetings_customer=[
            "Hey {name}, this is {agent} from {biz} — got a sec?",
            "Hi {name}, {agent} here from {biz}. Quick one, are you free to talk?",
            "Hey, it's {agent} from {biz}. {name}, am I catching you at a good time?",
        ],
        greetings_known_lead=[
            "Hi, is this {name}? It's {agent} from {biz} — quick one about {purpose}.",
            "Hey {name}, this is {agent} from {biz}. Got a minute? It's about {purpose}.",
        ],
        greetings_cold=[
            "Hi, this is {agent} from {biz} — calling about {purpose}. Who am I speaking with?",
            "Hey, {agent} here from {biz}. Got a moment? It's about {purpose}.",
        ],
        speak_instruction=(
            "Reply in natural conversational English. Indian English is fine "
            "if the caller speaks it — match their register."
        ),
        busy_handler_block=(
            "If they say 'make it fast', 'I have a meeting', 'I'm busy', DO NOT "
            "hang up. Acknowledge briefly, deliver the SPECIFIC reason in ONE "
            "sentence, then offer a callback or email summary."
        ),
        backchannels=["mmhmm.", "right.", "gotcha.", "okay.", "uh-huh.", "yeah."],
    ),

    # ──────────────────────────────────────────────────────────────────────
    "hi": LanguageProfile(
        code="hi",
        display_name="हिन्दी",
        # Cartesia "Hindi female, warm" — verified in their multilingual catalog.
        cartesia_voice_id="bf0a246a-8642-498a-9950-80c35e9276b5",
        elevenlabs_voice_id="zT03pEAEi0VHKciJODfn",
        greetings_customer=[
            "Namaste {name} ji, main {agent} bol raha hoon {biz} se — ek minute baat kar sakte hain?",
            "Hi {name} ji, {agent} hoon {biz} se. Aap free ho thodi der ke liye?",
            "Hello {name} ji, {agent} bol raha hoon {biz} ki taraf se. Sahi waqt hai baat karne ka?",
        ],
        greetings_known_lead=[
            "Namaste, kya yeh {name} ji hain? Main {agent} {biz} se — {purpose} ke baare mein ek minute.",
            "Hi {name} ji, {agent} hoon {biz} se. Ek minute hai aapke paas? {purpose} ke baare mein hai.",
        ],
        greetings_cold=[
            "Namaste, main {agent} {biz} se bol raha hoon — {purpose} ke baare mein call kar raha hoon. Aap kaun bol rahe hain?",
            "Hello, {agent} hoon {biz} se. Ek minute mil jayega? {purpose} ke baare mein hai.",
        ],
        speak_instruction=(
            "Reply in natural conversational Hindi. Use Hinglish where it "
            "sounds natural — Indian businesses talk this way. Use 'ji' "
            "for politeness with elders/customers. Don't translate brand "
            "names or proper nouns."
        ),
        busy_handler_block=(
            "Agar caller bole 'jaldi batao', 'meeting mein hoon', 'busy hoon' "
            "— call MAT katiye. Bas ek line mein specific reason boliye, fir "
            "poochiye 'callback chahiye ya email bhej doon?'."
        ),
        backchannels=["hmm.", "haan.", "theek hai.", "samajh gaya.", "achha."],
    ),

    # ──────────────────────────────────────────────────────────────────────
    "ta": LanguageProfile(
        code="ta",
        display_name="தமிழ்",
        cartesia_voice_id="b9de4a89-2257-424b-94c2-db18ba68c81a",
        elevenlabs_voice_id="zT03pEAEi0VHKciJODfn",
        greetings_customer=[
            "Vanakkam {name}, naan {agent} {biz} la irundhu pesuren — oru nimisham pesa mudiyuma?",
            "Hi {name}, {agent} pesuren {biz} la irundhu. Konjam free a irukingala?",
            "Hello {name}, {agent} dhan {biz} la irundhu. Sariyaana time a pesa?",
        ],
        greetings_known_lead=[
            "Vanakkam, {name} thaan a? Naan {agent} {biz} la irundhu — {purpose} pathi oru nimisham.",
            "Hi {name}, {agent} pesuren {biz} la irundhu. Konjam time iruka? {purpose} pathi.",
        ],
        greetings_cold=[
            "Vanakkam, naan {agent} {biz} la irundhu pesuren — {purpose} pathi call panren. Neenga yaaru pesureenga?",
            "Hello, {agent} dhan {biz} la irundhu. Konjam time kidaikuma? {purpose} pathi.",
        ],
        speak_instruction=(
            "Reply in natural conversational Tamil. Spoken Tamil with English "
            "loan words is fine — that's how Tamil businesses actually talk. "
            "Use 'neenga' (formal) for customers, not 'nee'."
        ),
        busy_handler_block=(
            "Caller 'seekram sollu', 'meeting la irukken', 'busy a irukken' "
            "sonna call vetiya VENDAM. Specific reason ah oru line la sollunga, "
            "appram 'callback venuma, email anuppattuma?' nu kelunga."
        ),
        backchannels=["hmm.", "aamaam.", "sari.", "puriyuthu."],
    ),

    # ──────────────────────────────────────────────────────────────────────
    "mr": LanguageProfile(
        code="mr",
        display_name="मराठी",
        cartesia_voice_id="03496517-369a-4db1-8236-3d3ae459ddf7",
        elevenlabs_voice_id="zT03pEAEi0VHKciJODfn",
        greetings_customer=[
            "Namaskar {name} ji, mi {agent} {biz} madhun bolto — ek minit bolu shakta ka?",
            "Hi {name}, {agent} bolto {biz} madhun. Free ahat ka thodavel?",
            "Namaskar {name}, {agent} ahe {biz} madhun. Yogya vela ahe ka bolayla?",
        ],
        greetings_known_lead=[
            "Namaskar, he {name} ahet ka? Mi {agent} {biz} madhun — {purpose} baddal ek minit.",
            "Hi {name}, {agent} bolto {biz} madhun. Ek minit ahe ka? {purpose} baddal ahe.",
        ],
        greetings_cold=[
            "Namaskar, mi {agent} {biz} madhun bolto — {purpose} baddal call karto ahe. Tumhi kon bolat ahat?",
            "Hello, {agent} ahe {biz} madhun. Ek minit milel ka? {purpose} baddal ahe.",
        ],
        speak_instruction=(
            "Reply in natural conversational Marathi. English loan words "
            "fine where they sound natural. Use respectful forms (tumhi, "
            "ahat) with customers, not casual (tu, ahes)."
        ),
        busy_handler_block=(
            "Caller 'lavkar sanga', 'meeting madhe ahe', 'busy ahe' bolla "
            "tar call kapu NAKA. Specific reason ek line madhe sanga, "
            "magh vichara 'callback pahije ka, email pathau ka?'."
        ),
        backchannels=["hmm.", "ho.", "barobar.", "samajla."],
    ),
}


def get_profile(language: str) -> LanguageProfile:
    """Return the language pack for a code, falling back to English."""
    code = (language or "en").strip().lower()
    return LANGUAGES.get(code) or LANGUAGES["en"]


def supported_languages() -> list[dict]:
    """For the frontend Settings picker."""
    return [
        {"code": p.code, "display_name": p.display_name}
        for p in LANGUAGES.values()
    ]
