"""
# LLM Report Generation

Generates radiology-style reports from deterministic imaging features (tumor volumes,
VASARI, midline shift). Runs locally using Ollama + Apptainer.

## Pipeline
1. Load subject metadata (quantitative features + paths).
2. Format into a structured JSON input block.
3. Generate narrative text using a selected LLM.
4. Save outputs as JSON, with automatic resume and split processing.

## Output Structure
Each subject produces:
- Findings (generated text)
- Original metadata

Results are stored under:
    reports/<MODEL_NAME>/brats23_metadata-report-<MODEL_NAME>-splitXofY.json

## Prompt Used
```python
'''
You are a radiologist generating a synthetic clinical MRI report.

Below is an example FINDINGS section taken from a real brain tumor report:

{example_findings}

Generate a similar FINDINGS section, **using only the metadata provided**.
Do not hallucinate measurements, anatomy, sequences, or findings not directly supported.
Write a concise, clinically coherent FINDINGS section grounded entirely in the structured fields.

Subject: {subject_id}
Metadata:
{metadata_json}
'''
```

## Usage
Run split 2 of 8:

    python generate_reports.py \
        --model gpt-oss:20b \
        --num_splits 8 \
        --split_no 2
"""
