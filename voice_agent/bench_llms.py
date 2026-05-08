"""
Benchmark Bedrock LLMs for voice-agent fitness.

Measures:
  - First-token latency (the only thing the caller actually feels)
  - Total response time
  - Tokens-per-second after first token
  - Cost per call (input+output × Bedrock pricing)
  - Reply quality (you eyeball it)

Run:
    cd nexuscaller-lab
    venv\\Scripts\\python -m voice_agent.bench_llms
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()


# Pricing per 1M tokens (Bedrock us-east-1, on-demand inference profile).
# Source: https://aws.amazon.com/bedrock/pricing/
PRICES_PER_MTOK = {
    # (input, output)
    "us.amazon.nova-micro-v1:0":          (0.035, 0.14),
    "us.amazon.nova-lite-v1:0":           (0.06,  0.24),
    "us.amazon.nova-pro-v1:0":            (0.80,  3.20),
    "us.meta.llama3-2-3b-instruct-v1:0":  (0.15,  0.15),
    "us.meta.llama3-1-8b-instruct-v1:0":  (0.22,  0.22),
    "us.meta.llama3-3-70b-instruct-v1:0": (0.72,  0.72),
    "mistral.mistral-7b-instruct-v0:2":   (0.15,  0.20),
    "mistral.mixtral-8x7b-instruct-v0:1": (0.45,  0.70),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0":  (0.80,  4.00),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.00, 15.00),
}

MODELS = list(PRICES_PER_MTOK.keys())


SYSTEM_PROMPT = (
    "You are Vox, a sales agent calling on a recorded line. "
    "Reply in ONE short sentence (max 15 words). Sound natural, like a human."
)
USER_TURN = "Yes, who is this calling?"


@dataclass
class Result:
    model: str
    ok: bool
    first_token_ms: int | None = None
    total_ms: int | None = None
    in_tokens: int = 0
    out_tokens: int = 0
    reply: str = ""
    error: str = ""

    @property
    def cost_usd(self) -> float:
        if self.model not in PRICES_PER_MTOK:
            return 0.0
        in_p, out_p = PRICES_PER_MTOK[self.model]
        return (self.in_tokens * in_p + self.out_tokens * out_p) / 1_000_000

    @property
    def tps(self) -> float:
        """Tokens per second after first token."""
        if not self.first_token_ms or not self.total_ms or self.out_tokens <= 1:
            return 0.0
        gen_ms = self.total_ms - self.first_token_ms
        if gen_ms <= 0:
            return 0.0
        return (self.out_tokens - 1) * 1000 / gen_ms


def _build_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(read_timeout=30, connect_timeout=10, retries={"max_attempts": 1}),
    )


def benchmark(client, model: str) -> Result:
    """Use Bedrock Converse Stream API — supported by Nova / Llama /
    Mistral / Anthropic with one API."""
    r = Result(model=model, ok=False)
    try:
        t0 = time.time()
        first_at = None
        out_text_parts = []
        in_tok = out_tok = 0

        resp = client.converse_stream(
            modelId=model,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": USER_TURN}]}],
            inferenceConfig={"maxTokens": 60, "temperature": 0.7},
        )

        for event in resp["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    if first_at is None:
                        first_at = time.time() - t0
                    out_text_parts.append(delta["text"])
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                in_tok = usage.get("inputTokens", 0)
                out_tok = usage.get("outputTokens", 0)

        total_at = time.time() - t0
        r.ok = True
        r.first_token_ms = int((first_at or total_at) * 1000)
        r.total_ms = int(total_at * 1000)
        r.in_tokens = in_tok
        r.out_tokens = out_tok
        r.reply = "".join(out_text_parts).strip()
    except Exception as e:
        r.error = str(e)[:120]
    return r


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("AWS_SECRET_ACCESS_KEY not set in .env — aborting.")
        sys.exit(1)

    client = _build_client()
    print(f"\nBenchmarking {len(MODELS)} models for voice-agent fitness…")
    print(f"Region: {os.getenv('AWS_REGION', 'us-east-1')}")
    print(f"User turn: {USER_TURN!r}")
    print(f"Expected reply: ~10-15 words\n")
    print(f"{'model':<55} {'first':<8} {'total':<8} {'tok/s':<7} {'$/turn':<10} {'reply'}")
    print(f"{'-' * 55} {'-' * 7:<8} {'-' * 7:<8} {'-' * 6:<7} {'-' * 9:<10} {'-' * 30}")

    results: list[Result] = []
    for model in MODELS:
        r = benchmark(client, model)
        results.append(r)
        if r.ok:
            print(f"{model:<55} {r.first_token_ms:>5}ms  {r.total_ms:>5}ms  {r.tps:>5.1f}  "
                  f"${r.cost_usd*1000:>6.4f}/k  {r.reply[:50]!r}")
        else:
            print(f"{model:<55} FAIL: {r.error}")

    print(f"\n{'-' * 100}\nLEADERBOARD (sorted by first-token latency)")
    ok_results = sorted([r for r in results if r.ok], key=lambda x: x.first_token_ms or 99999)
    print(f"{'rank':<5} {'model':<55} {'first':<10} {'$/10k turns'}")
    for i, r in enumerate(ok_results, 1):
        cost_10k = r.cost_usd * 10000
        print(f"{i:<5} {r.model:<55} {r.first_token_ms:>5}ms     ${cost_10k:>7.2f}")

    if ok_results:
        cheapest = min(ok_results, key=lambda x: x.cost_usd)
        fastest = ok_results[0]
        print(f"\n💰 Cheapest:  {cheapest.model}  →  ${cheapest.cost_usd*10000:.2f} per 10k turns")
        print(f"⚡ Fastest:   {fastest.model}  →  {fastest.first_token_ms}ms first token")
        # Sweet spot: best (cost × first_token) score
        scored = sorted(
            ok_results,
            key=lambda r: (r.cost_usd * 100_000) + (r.first_token_ms / 100),
        )
        print(f"🎯 Sweet spot:  {scored[0].model}  (best cost+latency balance)")


if __name__ == "__main__":
    main()
