import os, glob, shutil, json, argparse, re, logging
from .evaluation.utils import parse_radiology_report
from .evaluation.eval_tbfact import TBFactEval
from tqdm import tqdm
from .utils.log import get_logger
from RadEval import RadEval, compare_systems
from .utils import load_json
from pprint import pprint

def eval_json(args, include_keys=['BRAIN', 'MASS EFFECT & VENTRICLES',]):
    with open(args.json, "r") as f:
        data = json.load(f)

    for idx, (subject_id, metadata) in tqdm(enumerate(data.items()), total=len(data), colour='red'):
        print(f"[{idx}/{len(data)}] -- Evaluating: {subject_id}")

        save_path = args.json.replace('.json', '_eval_results.json')
        if args.do_details:
            save_path = save_path.replace('.json', "_details.json")

        existing_lookup = {}
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                existing_results = json.load(f)
            for d in existing_results:
                for sid, metrics in d.items():         
                    existing_lookup[str(sid)] = metrics

        # check that keys exist
        assert args.real_report_key in metadata, f"Missing real report key '{args.real_report_key}' in {args.json}"
        assert args.synthetic_report_key in metadata,  f"Missing synthetic report key '{args.synthetic_report_key}' in {args.json}"

        # # separate both real and generated reports into sections (e.g. FINDINGS, BRAIN, MASS EFFECT & VENTRICLES, etc.)
        real_reports = metadata[args.real_report_key]
        synthetic_reports = metadata[args.synthetic_report_key]
        refs =[real_reports]
        hyps = [synthetic_reports]
        if args.parse_real:
            real_reports = parse_radiology_report(real_reports)
            refs = ["\n\n".join(f"{k}:\n{real_reports[k]}" for k in include_keys if k in real_reports)]

        if args.parse_synthetic:
            synthetic_reports = parse_radiology_report(synthetic_reports)
            hyps = ["\n\n".join(f"{k}:\n{synthetic_reports[k]}" for k in include_keys if k in synthetic_reports)]

        pprint(refs)
        print('-*-*'*30)
        print('-*-*'*30)
        pprint(hyps)

        refs_missing = any(s.strip() == "" for s in refs)
        hyps_missing = any(s.strip() == "" for s in hyps)

        if refs_missing and hyps_missing:
            logger.info(f"Both reference and generated reports do not have {include_keys} sections")
            continue
            # return False, f"Both reference and generated reports do not have {include_keys} sections"
        elif refs_missing:
            logger.info(f"Reference report does not have {include_keys} sections")
            continue
            # return False, f"Reference report does not have {include_keys} sections"
        elif hyps_missing:
            logger.info(f"Generated report does not have {include_keys} sections")
            continue
            # return False, f"Generated report does not have {include_keys} sections"

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
            f'tbfact ({args.llm})': lambda hyps, refs: tbf_evaluator(refs=refs, hyps=hyps),  
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
            f'tbfact ({args.llm})': lambda hyps, refs: tbf_evaluator(refs=refs, hyps=hyps),  
            }

        # results = {str(subject_id): {k: fn(hyps, refs) for k, fn in eval_functions.items()}}
        subject_key = str(subject_id)
        prev = existing_lookup.get(subject_key, {})   # safe now

        current = {}

        for metric_name, fn in eval_functions.items():
            if (
                args.skip_processed
                and metric_name in prev
                and prev[metric_name] is not None
            ):
                logger.info(f"Skipping {metric_name} for {subject_key} (already computed)")
                current[metric_name] = prev[metric_name]
            else:
                logger.info(f" [{idx}/{len(data)}]-- Computing {metric_name} for {subject_key}")
                current[metric_name] = fn(hyps, refs)

        results = {subject_key: current}




        append_to_json_list(save_path, results)

        logger.info(f"Saved evaluation results to {save_path}")
        # return True, 'Success'




def append_to_json_list(save_path, new_item):
    """
    new_item must be a dict with exactly one key, e.g.:
        {"10064774423088": {...metrics...}}

    If the ID already exists in the JSON list:
        → merge the nested dicts (existing[id].update(new_data))
    Else:
        → append new_item to the list
    """

    # Load existing data
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of dicts.")
    else:
        data = []

    # Extract the patient ID from the new item
    if len(new_item) != 1:
        raise ValueError("new_item must contain exactly one top-level key.")
    new_id = list(new_item.keys())[0]
    new_value = new_item[new_id]

    # Check whether ID already exists and merge
    merged = False
    for entry in data:
        if new_id in entry:
            # Merge dictionaries (update existing metrics with new ones)
            entry[new_id].update(new_value)
            merged = True
            break

    # If not merged, append as new entry
    if not merged:
        data.append(new_item)

    # Save updated file
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    return data




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation across all subjects.")
    parser.add_argument("--json", type=str, required=True, help="Path to .json file, expected structure {'subject0_id':{'Clinical Report':.., 'Predicted Report':...}, ...}")
    parser.add_argument("--real_report_key", type=str, default='Clinical Report')
    parser.add_argument("--synthetic_report_key", type=str, default='Predicted Report')
    parser.add_argument("--devices", type=str, default='0,1,2,3')
    parser.add_argument("--do_details", action="store_true")
    parser.add_argument("--llm", type=str, default="deepseek-r1:70b")
    parser.add_argument("--skip_processed", action="store_true")


    parser.add_argument("--parse-real", action=argparse.BooleanOptionalAction, default=True, help="Parse the real clinical reports, use --no-parse-real for False",)
    parser.add_argument("--parse-synthetic", action=argparse.BooleanOptionalAction, default=True, help="Parse the synthetic generated reports, use --no-parse-synthetic for False",)


    args = parser.parse_args()


    logger = get_logger(args.json) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


    eval_json(args)
