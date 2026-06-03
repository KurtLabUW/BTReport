"""OpenAI-backed cluster summarization (parallel to collect_topics.describe_cluster_ollama)."""

from ..llm_report_generation.openai_client import DEFAULT_MODEL, chat


def describe_cluster_openai(facts, model=DEFAULT_MODEL, max_examples=8):
    """
    Produce a short semantic description of a cluster using the OpenAI API.
    Returns a fallback string if the API call fails.
    """
    sample_facts = facts[:max_examples]

    prompt = (
        "You are an expert radiologist summarizing clusters of radiology facts. "
        "Given the example fact statements below, write a short 3-8 word description "
        "that captures the shared semantic meaning.\n\n"
        "FACTS:\n- " + "\n- ".join(sample_facts) + "\n\nDescription:"
    )

    try:
        return chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        ).strip()
    except Exception as e:
        print(f"[WARN] OpenAI call failed: {e}")
        return "general finding category"
