# LLM Report Generation

Synthesizes structured radiology-style reports from quantitative features.

**Pipeline Overview:**
1. Aggregate deterministic features (metadata, VASARI, midline shift)
2. Format as structured input schema
3. Generate narrative text using a large language model (e.g., GPT, Llama, Mistral)

Reports are formatted in standardized sections:
- Indication  
- Technique  
- Findings  
- Impression

