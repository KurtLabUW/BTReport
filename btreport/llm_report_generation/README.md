# LLM Report Generation

Generates radiology-style FINDINGS sections from deterministic imaging features (tumor
volumes, VASARI, midline shift). Supports **OpenAI** (cloud API) and **Ollama** (local
via Apptainer).

Report generation routes models through `factory.py`.

## Pipeline

1. Load subject metadata (quantitative features + paths).
2. Format into a structured JSON input block.
3. Generate narrative text using a selected LLM.
4. Save outputs as JSON, with automatic resume and split processing.

## Model routing

| `--llm` example | Backend |
|-----------------|---------|
| `gpt-5.4-mini` (default) | OpenAI (name contains `gpt`, no `:`) |
| `llama3:70b`, `gpt-oss:120b` | Ollama (repo:tag with `:`) |
| `openai-gpt-5.4-mini` | OpenAI (explicit prefix) |
| `ollama-llama3:70b` | Ollama (explicit prefix) |
| `azure-gpt-5.4-mini` | OpenAI via `OPENAI_BASE_URL` |

## Setup

### OpenAI

```bash
pip install openai
export OPENAI_API_KEY=sk-...
# Optional: export OPENAI_BASE_URL=https://your-endpoint/v1

python -m btreport.openai_server check-api --model gpt-5.4-mini
```

### Ollama

See `../ollama_server.py` and `run_ollama_model.sh`. Start the server, then use a tagged
model name (e.g. `llama3:70b`, `gpt-oss:120b`).

## Usage

One entry point; backend is chosen from `--llm`:

```bash
# OpenAI (default model)
python -m btreport.generate_report \
  --subject_folder /path/to/subject \
  --llm gpt-5.4-mini

# Ollama
python -m btreport.generate_report \
  --subject_folder /path/to/subject \
  --llm llama3:70b
```

Programmatic:

```python
from btreport.llm_report_generation.factory import generate_llm_report, resolve_backend

resolve_backend("gpt-5.4-mini")  # -> "openai"
resolve_backend("llama3:70b")    # -> "ollama"

report = generate_llm_report(subject_id, metadata, model="gpt-5.4-mini")
report = generate_llm_report(subject_id, metadata, model="llama3:70b", prompt_version="v2")
```

Backend-specific modules (`openai_report_gen.py`, `ollama_report_gen.py`) remain available
for direct imports.

### Batch splits (Ollama)

For long subject lists, split work for array parallelization:

```bash
python ollama_report_gen.py \
  --model gpt-oss:120b \
  --num_splits 8 \
  --split_no 2
```

## Output structure

Each subject produces:

- Findings (generated text)
- Original metadata

Results are stored under:

```
reports/<MODEL_NAME>/brats23_metadata-report-<MODEL_NAME>-splitXofY.json
```

When using `generate_report`, the report is also saved in
`patient_metadata_btreport.json` under a key like
`BTReport Generated Report (<model>, run_name=<run_name>)`.

## Prompt

The v1 prompt template (shared by OpenAI and Ollama backends):

```python
'''
You are a radiologist generating a synthetic clinical MRI report.

Below is an example FINDINGS section taken from a real brain tumor report:

EXAMPLE FINDINGS:
{example_findings}

---

Now generate a similar FINDINGS section, **but ONLY using the metadata provided below**.
Do NOT mention measurements, structures, or features unless supported by the metadata.
Do NOT mention modalities or sequences not given.
The goal is to produce a realistic, readable FINDINGS section grounded purely in the structured fields.

METADATA (for subject {subject_id}):
{metadata_json}
---

Write the FINDINGS section now, using clinical radiology language.
'''
```


## Usage
For long subject lists, we split it to allow for array parallelization. The syntax is:

    python ollama_report_gen.py \
        --model gpt-oss:120b \
        --num_splits 8 \
        --split_no 2

## Report Evaluation
Reports generated with BTReport are compared to real clinical reports for each subject using RadEval. RadEval implements organ-agnostic radiology text-generation metrics including:
ROUGE-1 / ROUGE-2 / ROUGE-L; BERTScore; SRR-BERT; and RATEscore.

```python
from RadEval import RadEval
import json

evaluator = RadEval(
    do_rouge=True,
    do_bertscore=True,
    do_srr_bert=True,
    do_ratescore=True,
)

results = evaluator(refs=refs, hyps=hyps)
print(json.dumps(results, indent=2))
```
