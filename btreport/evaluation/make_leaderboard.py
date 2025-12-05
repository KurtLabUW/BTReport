import json, os, glob
import pandas as pd
from .utils import parse_radiology_report
from ..utils import load_json
from pprint import pprint
from RadEval import RadEval, compare_systems


EVAL_JSONS = {
    'BTReport (gpt-oss-120b)': '/pscratch/sd/j/jehr/MSFT/BTReport/data/example/merged_reports_btreport_eval_results_details.json',
    'BTReport (llama3-70b)': '/pscratch/sd/j/jehr/MSFT/BTReport/data/example/merged_reports_btreport_llama3_70b_eval_results_details.json',

    'AutoRG-Brain': "/pscratch/sd/j/jehr/MSFT/BTReport_evaluation/AutoRG_Brain_weights/autorg_reports_uwimaging_eval_results_details.json",
    'seg2exp': "/pscratch/sd/j/jehr/MSFT/BTReport_evaluation/from-segmentation-to-explanation/saved/seg2exp_reports_uwimaging_eval_results_details.json",
}

REPORT_JSONS_DICT = {
    'BTReport (gpt-oss-120b)':{'path':'/pscratch/sd/j/jehr/MSFT/BTReport/data/example/merged_reports_btreport.json', 'parse_real': True, 'parse_synthetic':True,},
    'BTReport (llama3-70b)':{'path':'/pscratch/sd/j/jehr/MSFT/BTReport/data/example/merged_reports_btreport_llama3_70b.json', 'parse_real': True, 'parse_synthetic':True,},

    'AutoRG-Brain':{'path': '/pscratch/sd/j/jehr/MSFT/BTReport_evaluation/AutoRG_Brain_weights/autorg_reports_uwimaging.json', 'parse_real': True, 'parse_synthetic':False,},
    'seg2exp':{'path': '/pscratch/sd/j/jehr/MSFT/BTReport_evaluation/from-segmentation-to-explanation/saved/seg2exp_reports_uwimaging.json', 'parse_real': True, 'parse_synthetic':False,}

}


def significance_test(systems, references):
    rouge_evaluator = RadEval(do_rouge=True)
    bleu_evaluator = RadEval(do_bleu=True)
    bertscore_evaluator = RadEval(do_bertscore=True)
    ratescore_evaluator = RadEval(do_ratescore=True)

    eval_functions = {
    'bleu': lambda hyps, refs: bleu_evaluator(refs, hyps)['bleu'],
    'rouge1': lambda hyps, refs: rouge_evaluator(refs, hyps)['rouge1'],
    'rouge2': lambda hyps, refs: rouge_evaluator(refs, hyps)['rouge2'],
    'rougeL': lambda hyps, refs: rouge_evaluator(refs, hyps)['rougeL'],
    'bertscore': lambda hyps, refs: bertscore_evaluator(refs, hyps)['bertscore'],
    'ratescore': lambda hyps, refs: ratescore_evaluator(refs, hyps)['ratescore'],
    }

    signatures, scores = compare_systems(
        systems=systems,
        metrics=eval_functions,
        references=references,
        n_samples=50,                    # Number of randomization samples
        significance_level=0.05,         # Alpha level for significance testing
        print_results=True              # Print formatted results table
    )
    return signatures, scores


def process_eval_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    for entry in data:
        # each entry is {"subject_id": {metrics}}
        subject_id, metrics = next(iter(entry.items()))

        bleu = metrics.get("bleu", {}).get("bleu", {})
        rouge = metrics.get('rouge1', {}).get('rouge', {})
        bertscore = metrics.get('bertscore', {}).get('bertscore', {})
        ratescore = metrics.get('ratescore', {}).get('ratescore', {})

        tbfact_deepseek = metrics.get("tbfact (deepseek-r1:70b)", {})


        rows.append({
            "subject_id": subject_id,
            "bleu_1": bleu.get("bleu_1"),
            "bleu_2": bleu.get("bleu_2"),
            "bleu_3": bleu.get("bleu_3"),
            "bleu_4": bleu.get("bleu_4"),

            "rouge_1": rouge.get('rouge1').get('mean_score'),
            "rouge_2": rouge.get('rouge2').get('mean_score'),
            "rouge_L": rouge.get('rougeL').get('f1-score'),

            "bertscore": bertscore.get('mean_score'),
            "ratescore": ratescore.get('f1-score'),

            
            "tbfact_deepseek": tbfact_deepseek.get('score'),
            "tbfact_deepseek_precision": tbfact_deepseek.get('details', {}).get('metrics', {}).get('precision'),
            "tbfact_deepseek_recall": tbfact_deepseek.get('details', {}).get('metrics', {}).get('recall'),
            "tbfact_deepseek_f1": tbfact_deepseek.get('details', {}).get('metrics', {}).get('f1'),


        })

    return pd.DataFrame(rows)





def print_eval_metrics(eval_jsons_dict=EVAL_JSONS):
    results_summary = {}
    for name, eval_json in eval_jsons_dict.items():
        eval_df = process_eval_json(eval_json)

        numeric_df = eval_df.drop(columns=["subject_id"], errors="ignore")
        numeric_df = numeric_df.select_dtypes(include="number")
        mean_metrics_df =  numeric_df.mean().to_dict()

        results_summary[name] = mean_metrics_df

    print(json.dumps(results_summary, indent=2))


def print_significance_tests(REPORT_JSONS_DICT):
    """
    REPORT_JSONS_DICT: dict { model_name: path_to_json }
    
    Returns:
        systems: dict of model_name -> list of predicted reports
        references: list of clinical (ground truth) reports
    """

    # Load all JSON files
    model_data = {}
    for model, report in REPORT_JSONS_DICT.items():
        with open(report['path'], "r") as f:
            model_data[model] = json.load(f)

    # Find subject IDs that are present in ALL models
    subject_sets = [set(d.keys()) for d in model_data.values()]
    common_subjects = sorted(set.intersection(*subject_sets))

    if len(common_subjects) == 0:
        raise ValueError("No overlapping subject IDs across all models.")

    # Build the output structures
    systems = {model: [] for model in REPORT_JSONS_DICT}
    references = []

    for sid in common_subjects:
        # Reference: same for all models → from any model
        model_ = next(iter(REPORT_JSONS_DICT))
        
        conf = REPORT_JSONS_DICT[model_]
        subdict = model_data[model_][sid]

        ref = subdict["Clinical Report"]
        if conf["parse_real"]:
            real_reports = parse_radiology_report(ref)
            ref = " ".join(f"{k}:\n{real_reports[k]}" for k in ['BRAIN', 'MASS EFFECT & VENTRICLES',] if k in real_reports)

        references.append(ref)

        # Predictions per model
        for model in REPORT_JSONS_DICT:
            subdict = model_data[model][sid]
            pred = subdict["Predicted Report"]
            conf = REPORT_JSONS_DICT[model]

            if conf["parse_synthetic"]:
                synthetic_reports = parse_radiology_report(pred)
                pred = "\n\n".join(f"{k}:\n{synthetic_reports[k]}" for k in ['BRAIN', 'MASS EFFECT & VENTRICLES',] if k in synthetic_reports)

            systems[model].append(pred)

    def preview(s, n=100):
        return s[:n].replace("\n", " ") + ("..." if len(s) > n else "")

    print("\n=== REFERENCES PREVIEW ===")
    for i, r in enumerate(references[:3]):
        print(f"ref[{i}]:", preview(r))

    print("\n=== SYSTEMS PREVIEW ===")
    for model in systems:
        print(f"\n-- {model} --")
        for i, hyp in enumerate(systems[model][:3]):
            print(f"hyp[{i}]:", preview(hyp))

    significance_test(systems, references)

# TODO: add parsing foloowing REPORT_JSONS_DICT


if __name__ == '__main__':
    print_eval_metrics()
    # print_significance_tests(REPORT_JSONS_DICT)