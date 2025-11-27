import os, argparse, asyncio
import json
from typing import Dict, Any
from .tbfact import LLMClient, TBFactEvaluator
from .utils import parse_radiology_report

from ollama import chat   
from ..utils.log import get_logger


class OllamaLLMClient(LLMClient):
    """
    Drop-in replacement for MedBenchLLMClientAdapter
    Uses ollama.chat. No Azure, no Runner, no SystemPromptModel, etc
    """

    def __init__(self, model: str = "llama3"):
        self.model = model   # e.g.: "qwen:7b", "gpt-oss:120b"

    async def generate(self, system_message: str, user_message: str) -> Dict[str, Any]:
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ]
        )
        text = response["message"]["content"]
        return {"content": text}



async def eval_single_subject(args, include_keys=['BRAIN', 'MASS EFFECT & VENTRICLES',]):
    metadata_json_pth=os.path.join(args.subject_folder, 'patient_metadata_btreport.json')
    with open(metadata_json_pth, 'r') as f:
        metadata = json.load(f)

    assert args.real_report_key in metadata, f"Missing real report key '{args.real_report_key}' in {metadata_json_pth}"
    assert args.synthetic_report_key in metadata,  f"Missing synthetic report key '{args.synthetic_report_key}' in {metadata_json_pth}"

    real_reports = parse_radiology_report(metadata[args.real_report_key])
    synthetic_reports = parse_radiology_report(metadata[args.synthetic_report_key])

    real_reports = "\n\n".join(f"{k}:\n{real_reports[k]}" for k in include_keys if k in real_reports)
    synthetic_reports = "\n\n".join(f"{k}:\n{synthetic_reports[k]}" for k in include_keys if k in synthetic_reports)

    tbfact_evaluator = TBFactEvaluator(llm_client=OllamaLLMClient(model = args.llm))
    scores = await tbfact_evaluator.evaluate(generated_text=synthetic_reports, reference_text=real_reports)
    print(json.dumps(scores, indent=2))
    return scores



class TBFactEval:
    def __init__(self, llm="gpt-oss:120b", do_details=True):
        self.llm = llm
        self.do_details = do_details

    def __call__(self, refs, hyps):
        evaluator = TBFactEvaluator(llm_client=OllamaLLMClient(model=self.llm))
        result = asyncio.run(evaluator.evaluate(generated_text=hyps, reference_text=refs))
        return result if self.do_details else result["score"]


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Generate a brain tumor report for one subject.")
    parser.add_argument("--subject_folder", type=str, help="Path to the subject folder containing the MRI data.", required=True)
    parser.add_argument("--real_report_key", type=str, default='Clinical Report')
    parser.add_argument("--synthetic_report_key", type=str, default='llm report')
    parser.add_argument("--devices", type=str, default='0,1,2,3', help="String with cuda device IDs for use by synthseg and SynthMorph. E.g. '0,1' or '0'.")
    parser.add_argument("--llm", type=str, default="gpt-oss:120b")


    args = parser.parse_args()

    subject = os.path.basename(os.path.normpath(args.subject_folder))
    logger = get_logger(subject) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    asyncio.run(eval_single_subject(args))