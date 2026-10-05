"""Gemini API integration (google-genai SDK): prompt construction and generation with retry."""

import os
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


def format_context(chunks: pd.DataFrame) -> str:
    """Label each excerpt with its section and page so the model can cite it."""
    return "\n\n".join(
        f"[Section: {row.section} | Page {row.page}]\n{row.text}"
        for row in chunks.itertuples()
    )


def build_prompt(prompts: dict, question: str, brand: str, model: str, chunks: pd.DataFrame) -> str:
    """Fill the user_template from the prompt file."""
    if chunks.empty:
        raise ValueError("Refusing to build a prompt with no retrieved context.")
    return prompts["user_template"].format(
        brand=brand,
        model=model,
        context=format_context(chunks),
        question=question,
    )


class Generator:
    """Thin wrapper around the Gemini API."""

    def __init__(self, cfg: dict, prompts: dict):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not found. Copy .env.example to .env and add your key."
            )

        gen_cfg = cfg["generation"]
        self.max_retries = gen_cfg["max_retries"]
        self.retry_delay = gen_cfg["retry_delay_s"]
        self.model_name = cfg["models"]["llm"]

        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            system_instruction=prompts["system"],
            temperature=gen_cfg["temperature"],
            max_output_tokens=gen_cfg["max_output_tokens"],
        )

    def generate(self, prompt: str) -> tuple[str, str]:
        """Return (answer, status). Retries only on rate-limit errors, with backoff."""
        delay = self.retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.config,
                )
                text = (response.text or "").strip()
                if not text:
                    # empty or blocked by safety filters
                    return "The model returned no answer for this question. Try rephrasing it.", "empty_response"
                return text, "success"
            except errors.APIError as exc:
                rate_limited = exc.code == 429
                if rate_limited and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if rate_limited:
                    return "Gemini API quota exceeded. Please wait a minute and try again.", "rate_limited"
                return f"Generation failed: {exc}", "error"
            except Exception as exc:
                return f"Generation failed: {exc}", "error"
        return "Generation failed.", "error"