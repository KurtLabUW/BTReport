import json

from sanitext.text_sanitization import sanitize_text

from .openai_client import DEFAULT_MODEL, chat
from .ollama_report_gen import (
    EXAMPLE_FINDINGS,
    REPORT_TEMPLATE,
    REPORT_TEMPLATE_IMAGE,
)


def generate_llm_report(subject_id, metadata, image_path=None, model=DEFAULT_MODEL):
    if image_path is None:
        prompt = REPORT_TEMPLATE.format(
            example_findings=EXAMPLE_FINDINGS,
            subject_id=subject_id,
            metadata_json=json.dumps(metadata, indent=2),
        )
    else:
        prompt = REPORT_TEMPLATE_IMAGE.format(
            example_findings=EXAMPLE_FINDINGS,
            image_path=image_path,
            subject_id=subject_id,
            metadata_json=json.dumps(metadata, indent=2),
        )

    report = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        image_path=image_path,
    )
    report = report.replace("\u2011", "-")
    return sanitize_text(report)
