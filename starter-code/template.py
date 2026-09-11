"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI
    from time import time
    start_time = time()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    end_time = time()
    latency_seconds = end_time - start_time
    response_text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }
    return response_text, latency_seconds, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types
    from time import time
    start_time = time()
    response = genai.generate_content(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    end_time = time()
    latency_seconds = end_time - start_time
    response_text = response.output_text
    usage = {
        "input_tokens": response.usage_metadata.input_tokens,
        "output_tokens": response.usage_metadata.output_tokens
    }
    return response_text, latency_seconds, usage
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    raise NotImplementedError("Implement call_gemini")


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic
    from time import time
    start_time = time()
    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens_to_sample=max_tokens,
    )
    end_time = time()
    latency_seconds = end_time - start_time
    response_text = response.completion
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }
    return response_text, latency_seconds, usage
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.
    raise NotImplementedError("Implement call_anthropic")


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    gpt4o_response, gpt4o_latency, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_mini_response, gpt4o_mini_latency, gpt4o_mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_flash_response, gemini_flash_latency, gemini_flash_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gpt4o_cost = (gpt4o_usage['input_tokens'] * PRICING_1M_TOKENS[OPENAI_MODEL]['input'] + gpt4o_usage['output_tokens'] * PRICING_1M_TOKENS[OPENAI_MODEL]['output']) / 1_000_000
    gpt4o_mini_cost = (gpt4o_mini_usage['input_tokens'] * PRICING_1M_TOKENS[OPENAI_MINI_MODEL]['input'] + gpt4o_mini_usage['output_tokens'] * PRICING_1M_TOKENS[OPENAI_MINI_MODEL]['output']) / 1_000_000
    gemini_flash_cost = (gemini_flash_usage['input_tokens'] * PRICING_1M_TOKENS[GEMINI_MODEL]['input'] + gemini_flash_usage['output_tokens'] * PRICING_1M_TOKENS[GEMINI_MODEL]['output']) / 1_000_000
    return {
        "gpt4o": {
            "response": gpt4o_response,
            "latency": gpt4o_latency,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage['input_tokens'],
            "output_tokens": gpt4o_usage['output_tokens'],
        },
        "gpt4o_mini": {
            "response": gpt4o_mini_response,
            "latency": gpt4o_mini_latency,
            "cost": gpt4o_mini_cost,
            "input_tokens": gpt4o_mini_usage['input_tokens'],
            "output_tokens": gpt4o_mini_usage['output_tokens'],
        },
        "gemini_flash": {
            "response": gemini_flash_response,
            "latency": gemini_flash_latency,
            "cost": gemini_flash_cost,
            "input_tokens": gemini_flash_usage['input_tokens'],
            "output_tokens": gemini_flash_usage['output_tokens'],
        },
    }
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.
    raise NotImplementedError("Implement compare_models")


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[dict[str, Any]] = []

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "parts": [{"text": user_input}]})
        response_parts = []
        print("Gemini: ", end="", flush=True)
        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history,
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                response_parts.append(chunk.text)
        print()

        history.append(
            {"role": "model", "parts": [{"text": "".join(response_parts)}]}
        )
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2**attempt)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        comparison["prompt"] = prompt
        results.append(comparison)
    return results



# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        prompt = str(result["prompt"]).replace("\n", " ").replace("|", "\\|")
        for model, stats in result.items():
            if model == "prompt":
                continue
            response = str(stats["response"]).replace("\n", " ").replace("|", "\\|")[:50]
            rows.append(
                f"| {prompt} | {model_names.get(model, model)} | {response} | "
                f"{stats['latency']:.2f}s | {stats['input_tokens']}/{stats['output_tokens']} | "
                f"${stats['cost']:.6f} |"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
