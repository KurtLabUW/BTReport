"""
Route model strings to OpenAI or Ollama report generators.

Use explicit prefixes to override heuristics:
  openai-gpt-5.4-mini    -> OpenAI
  ollama-llama3:70b      -> Ollama

Without a prefix, routing uses naming conventions:
  - Names with a tag (colon), e.g. llama3:70b, gpt-oss:120b -> Ollama
  - Names containing gpt, o1, o3, or o4 without a colon -> OpenAI
"""

from typing import Callable, Tuple

Backend = str  # "openai" | "ollama"

_OPENAI_HINTS = ("gpt", "o1", "o3", "o4")


def resolve_backend(model_string: str) -> Backend:
    """Return which LLM backend should handle this model string."""
    backend, _ = normalize_model_string(model_string)
    return backend


def normalize_model_string(model_string: str) -> Tuple[Backend, str]:
    """
    Resolve backend and return the model name passed to that backend's API.

    Strips routing prefixes (openai-, ollama-).
    """
    model = model_string.strip()
    if not model:
        raise ValueError("Model string cannot be empty.")

    lower = model.lower()
    if lower.startswith("openai-"):
        return "openai", model[7:]
    if lower.startswith("ollama-"):
        return "ollama", model[7:]

    # Ollama models use repo:tag (e.g. llama3:70b, gpt-oss:120b).
    if ":" in model:
        return "ollama", model

    if any(hint in lower for hint in _OPENAI_HINTS):
        return "openai", model

    raise ValueError(
        f"Cannot infer backend for model {model_string!r}. "
        "Use an explicit prefix: openai-<model> or ollama-<model>. "
        "Examples: openai-gpt-5.4-mini, ollama-llama3:70b, gpt-5.4-mini, deepseek-r1:70b."
    )


def _load_generate_llm_report(backend: Backend, prompt_version: str) -> Callable:
    if prompt_version not in ("v1", "v2"):
        raise ValueError(f"prompt_version must be 'v1' or 'v2', got {prompt_version!r}")

    if backend == "openai":
        if prompt_version == "v2":
            from .openai_report_gen_v2 import generate_llm_report
        else:
            from .openai_report_gen import generate_llm_report
    else:
        if prompt_version == "v2":
            from .ollama_report_gen_v2 import generate_llm_report
        else:
            from .ollama_report_gen import generate_llm_report
    return generate_llm_report


def generate_llm_report(
    subject_id,
    metadata,
    image_path=None,
    model: str = "gpt-5.4-mini",
    *,
    prompt_version: str = "v1",
    **kwargs,
):
    """
    Generate a FINDINGS report using the backend inferred from ``model``.

    Same signature as openai_report_gen / ollama_report_gen implementations.
    """
    backend, normalized_model = normalize_model_string(model)
    impl = _load_generate_llm_report(backend, prompt_version)
    return impl(
        subject_id,
        metadata,
        image_path=image_path,
        model=normalized_model,
        **kwargs,
    )
