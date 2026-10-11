"""Optional local Ollama integration using only the Python standard library."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_ENDPOINT = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def build_prompt(
    product: dict[str, Any],
    probabilities: dict[str, float],
    model_factors: list[dict[str, Any]],
) -> str:
    """Build an Indonesian, evidence-grounded product analysis prompt."""
    product_facts = {
        key: product.get(key)
        for key in (
            "product_name",
            "main_category",
            "category",
            "discounted_price",
            "actual_price",
            "discount_percentage",
            "about_product",
        )
    }
    evidence = {
        "product": product_facts,
        "model_probabilities": probabilities,
        "global_feature_importance": model_factors,
    }
    return (
        "Anda adalah analis bisnis yang membantu menafsirkan data produk Amazon. "
        "Gunakan hanya fakta pada JSON berikut. Probabilitas adalah prediksi "
        "Rating_Class, bukan penyebab atau kepastian; feature importance global "
        "bukan penjelasan kausal produk ini. Jangan mengarang fakta.\n\n"
        f"DATA:\n{json.dumps(evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Tulis ringkasan singkat dalam bahasa Indonesia berisi kelas rating yang "
        "paling mungkin, dua observasi yang merujuk langsung pada data, satu "
        "saran bisnis yang proporsional, dan satu batasan interpretasi."
    )


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: int = 60,
) -> str:
    """Call a local Ollama chat endpoint; raise explicit errors on failure."""
    request_body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    request = Request(
        f"{endpoint}/api/chat",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama mengembalikan HTTP {error.code}: {details}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Tidak dapat terhubung ke Ollama lokal ({endpoint}): {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Respons Ollama bukan JSON yang valid.") from error

    content = payload.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Respons Ollama tidak berisi teks jawaban.")
    return content.strip()
