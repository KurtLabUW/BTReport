"""OpenAI Chat Completions client for BTReport."""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "gpt-5.4-mini"


def check_env_variables() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY. Syntax: export OPENAI_API_KEY=sk-..."
        )


def get_client():
    check_env_variables()
    from openai import OpenAI

    kwargs: Dict[str, Any] = {"api_key": os.environ["OPENAI_API_KEY"]}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _image_content_part(image_path: str) -> Dict[str, Any]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")

    encoded = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    image_path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Run a chat completion and return assistant text.

    If image_path is set, the last user message is sent as multimodal content
    (text + image) for vision-capable models.
    """
    client = get_client()
    request_messages = [dict(m) for m in messages]

    if image_path:
        if not request_messages or request_messages[-1].get("role") != "user":
            raise ValueError("image_path requires the last message to be from the user")
        user_text = request_messages[-1].get("content", "")
        if not isinstance(user_text, str):
            raise ValueError("image_path requires string user message content")
        request_messages[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                _image_content_part(image_path),
            ],
        }

    response = client.chat.completions.create(
        model=model,
        messages=request_messages,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def check_api(model: str = DEFAULT_MODEL) -> bool:
    """Verify API credentials with a minimal completion."""
    text = chat(
        model=model,
        messages=[{"role": "user", "content": "Reply with OK."}],
    )
    print(f"OpenAI API reachable (model={model}, reply={text!r})")
    return True
