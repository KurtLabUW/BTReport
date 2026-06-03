"""CLI helpers for OpenAI API access (no local LLM server required)."""

import argparse

from .llm_report_generation.openai_client import DEFAULT_MODEL, check_api, check_env_variables


def main():
    p = argparse.ArgumentParser(description="Verify OpenAI API access for BTReport.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check-api", help="Run a minimal API call to verify credentials")
    p_check.add_argument("--model", default=DEFAULT_MODEL)

    args = p.parse_args()

    if args.cmd == "check-api":
        check_env_variables()
        check_api(model=args.model)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
