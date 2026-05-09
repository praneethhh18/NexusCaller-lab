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
    "us.meta.llama3-1-8b-instruct-v1:0":  (0.22,  0.22),
    "us.meta.llama3-3-70b-instruct-v1:0": (0.72,  0.72),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0":  (0.80,  4.00),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.00, 15.00),
    "ai21.jamba-1-5-mini-v1:0":           (0.20,  0.40),
    "ai21.jamba-1-5-large-v1:0":          (2.00,  8.00),
    # additional candidates to evaluate
    "cohere.command-r-v1:0":              (0.50,  1.50),
    "cohere.command-r-plus-v1:0":         (3.00, 15.00),
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
    print("\nWarm benchmark: 1 warm-up + 3 measured runs averaged.")
    print(f"Region: {os.getenv('AWS_REGION', 'us-east-1')}")
    print(f"User turn: {USER_TURN!r}\n")
    print(f"{'model':<55} {'avg first':<11} {'best':<8} {'$/10k':<10}")
    print(f"{'-' * 55} {'-' * 10:<11} {'-' * 7:<8} {'-' * 9}")

    results: list[Result] = []
    for model in MODELS:
        # 1 warm-up call (don't count cold-start)
        warm = benchmark(client, model)
        if not warm.ok:
            print(f"{model:<55} FAIL: {warm.error[:80]}")
            results.append(warm)
            continue

        # 3 measured runs
        runs = [benchmark(client, model) for _ in range(3)]
        runs = [r for r in runs if r.ok]
        if not runs:
            print(f"{model:<55} FAIL on warm runs")
            continue
        avg_first = sum(r.first_token_ms for r in runs) // len(runs)
        best_first = min(r.first_token_ms for r in runs)
        avg_in = sum(r.in_tokens for r in runs) / len(runs)
        avg_out = sum(r.out_tokens for r in runs) / len(runs)
        # synthetic "result" for the leaderboard
        agg = Result(
            model=model, ok=True,
            first_token_ms=avg_first,
            total_ms=sum(r.total_ms for r in runs) // len(runs),
            in_tokens=int(avg_in), out_tokens=int(avg_out),
            reply=runs[0].reply,
        )
        results.append(agg)
        print(f"{model:<55} {avg_first:>5}ms     {best_first:>5}ms   ${agg.cost_usd*10000:>7.2f}")

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
