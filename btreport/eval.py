import os, glob, shutil, json, argparse, re, logging
from .evaluation.utils import parse_radiology_report
from .evaluation.eval_tbfact import TBFactEval
from .utils.log import get_logger
from RadEval import RadEval
from .utils import load_json
from pprint import pprint

# from loguru import logger
# logger.remove()
logger = logging.getLogger(__name__)


def eval_single_subject(args, include_keys=['BRAIN', 'MASS EFFECT & VENTRICLES',]):
    if args.subject_folder:
        metadata_json_pth=os.path.join(args.subject_folder, 'patient_metadata_btreport.json')
        metadata = load_json(metadata_json_pth)
    else:
        metadata_json_pth=args.json
        metadata = load_json(metadata_json_pth)

    # check that keys exist
    assert args.real_report_key in metadata, f"Missing real report key '{args.real_report_key}' in {metadata_json_pth}"
    assert args.synthetic_report_key in metadata,  f"Missing synthetic report key '{args.synthetic_report_key}' in {metadata_json_pth}"

    # # separate both real and generated reports into sections (e.g. FINDINGS, BRAIN, MASS EFFECT & VENTRICLES, etc.)
    real_reports = metadata[args.real_report_key]
    synthetic_reports = metadata[args.synthetic_report_key]
    if args.parse_real:
        real_reports = parse_radiology_report(real_reports)
    if args.parse_synthetic:
        synthetic_reports = parse_radiology_report(synthetic_reports)

    pprint(real_reports)
    print('-*-*'*30)
    print('-*-*'*30)
    pprint(synthetic_reports)


    # # select subsections from report, merge them into one text block
    refs = ["\n\n".join(f"{k}:\n{real_reports[k]}" for k in include_keys if k in real_reports)]
    hyps = ["\n\n".join(f"{k}:\n{synthetic_reports[k]}" for k in include_keys if k in synthetic_reports)]

    refs_missing = any(s.strip() == "" for s in refs)
    hyps_missing = any(s.strip() == "" for s in hyps)

    if refs_missing and hyps_missing:
        logger.info(f"Both reference and generated reports do not have {include_keys} sections")
        return False, f"Both reference and generated reports do not have {include_keys} sections"
    elif refs_missing:
        logger.info(f"Reference report does not have {include_keys} sections")
        return False, f"Reference report does not have {include_keys} sections"
    elif hyps_missing:
        logger.info(f"Generated report does not have {include_keys} sections")
        return False, f"Generated report does not have {include_keys} sections"


    # print(refs)
    # print('-*-*'*30)
    # print('-*-*'*30)
    # print(hyps)

    rouge_evaluator = RadEval(do_rouge=True, do_details=args.do_details)
    bleu_evaluator = RadEval(do_bleu=True, do_details=args.do_details)
    bertscore_evaluator = RadEval(do_bertscore=True, do_details=args.do_details)
    srr_bert_evaluator = RadEval(do_radgraph=True, do_details=args.do_details)
    ratescore_evaluator = RadEval(do_ratescore=True, do_details=args.do_details)
    tbf_evaluator = TBFactEval(llm=args.llm, do_details=args.do_details)

    if not args.do_details:
        eval_functions = {
        'bleu': lambda hyps, refs: bleu_evaluator(refs, hyps)['bleu'],
        'rouge1': lambda hyps, refs: rouge_evaluator(refs, hyps)['rouge1'],
        'rouge2': lambda hyps, refs: rouge_evaluator(refs, hyps)['rouge2'],
        'rougeL': lambda hyps, refs: rouge_evaluator(refs, hyps)['rougeL'],
        'bertscore': lambda hyps, refs: bertscore_evaluator(refs, hyps)['bertscore'],
        # 'radgraph': lambda hyps, refs: srr_bert_evaluator(refs, hyps)['radgraph_partial'],
        'ratescore': lambda hyps, refs: ratescore_evaluator(refs, hyps)['ratescore'],
        'tbfact (args.llm)': lambda hyps, refs: tbf_evaluator(refs=refs, hyps=hyps),  
        }
    else:
        eval_functions = {
        'bleu': lambda hyps, refs: bleu_evaluator(refs, hyps),
        'rouge1': lambda hyps, refs: rouge_evaluator(refs, hyps),
        'rouge2': lambda hyps, refs: rouge_evaluator(refs, hyps),
        'rougeL': lambda hyps, refs: rouge_evaluator(refs, hyps),
        'bertscore': lambda hyps, refs: bertscore_evaluator(refs, hyps),
        # 'radgraph': lambda hyps, refs: srr_bert_evaluator(refs, hyps),
        'ratescore': lambda hyps, refs: ratescore_evaluator(refs, hyps),
        'tbfact (args.llm)': lambda hyps, refs: tbf_evaluator(refs=refs, hyps=hyps),  
        }

    results = {k: fn(hyps, refs) for k, fn in eval_functions.items()}

    save_path = os.path.join(args.subject_folder, "eval_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved evaluation results to {save_path}")
    return True, 'Success'


    '''
    {
    "bertscore": 0.37174591422080994,
    "rouge1": 0.343801652892562,
    "rouge2": 0.08623548922056384,
    "rougeL": 0.15206611570247935,
    "srr_bert_weighted_f1": 0.75,
    "srr_bert_weighted_precision": 0.75,
    "srr_bert_weighted_recall": 0.75,
    "ratescore": 0.5243761568144104
    }


    
    '''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a brain tumor report for one subject.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject_folder", type=str, help="Path to the subject folder containing the BTReport results .json file.")
    group.add_argument("--json", type=str, help="Path to .json file, expected structure {'subject0_id':{real_report_key:.., synthetic_report_key:...}, ...}")    
    
    parser.add_argument("--real_report_key", type=str, default='Clinical Report')
    parser.add_argument("--synthetic_report_key", type=str, default='BTReport Generated Report (gpt-oss:120b)')
    parser.add_argument("--devices", type=str, default='0,1,2,3', help="String with cuda device IDs for use by synthseg and SynthMorph. E.g. '0,1' or '0'.")
    parser.add_argument("--do_details", action="store_true")
    parser.add_argument("--llm", type=str, default="gpt-oss:120b", help="LLM to be used by TBFact evaluator.")

    parser.add_argument("--parse-real", action=argparse.BooleanOptionalAction, default=True, help="Parse the real clinical reports, use --no-parse-real for False",)
    parser.add_argument("--parse-synthetic", action=argparse.BooleanOptionalAction, default=True, help="Parse the synthetic generated reports, use --no-parse-synthetic for False",)


    args = parser.parse_args()

    subject = os.path.basename(os.path.normpath(args.subject_folder))
    logger = get_logger(subject) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    eval_single_subject(args)
