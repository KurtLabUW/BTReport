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

