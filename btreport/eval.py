import os, glob, shutil, json, argparse, re
from .utils.log import get_logger
from RadEval import RadEval
from pprint import pprint

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


TOP_LEVEL = [
    "CLINICAL INDICATION",
    "TECHNIQUE",
    "CONTRAST",
    "COMPARISON",
    "FINDINGS",
    "HEAD MRA",
    "IMPRESSION"
]

SUBFINDINGS = [
    "MASS EFFECT & VENTRICLES",
    "BRAIN",
    "ENHANCEMENT",
    "VASCULAR",
    "EXTRA-AXIAL",
    "EXTRA-CRANIAL"
]

def parse_radiology_report(text):

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    header_regex = r"(?im)^(" + "|".join(TOP_LEVEL) + r")\s*:?\s*$"
    matches = list(re.finditer(header_regex, text))

    sections = {}
    for i,m in enumerate(matches):
        key = m.group(1).strip().upper()
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()

    if "FINDINGS" in sections:
        block = sections["FINDINGS"]

        sub_regex = r"(?im)^(" + "|".join(SUBFINDINGS) + r")\s*:?\s*"
        sub = list(re.finditer(sub_regex, block))

        for i,m in enumerate(sub):
            key = m.group(1).strip().upper()
            start = m.end()
            end   = sub[i+1].start() if i+1 < len(sub) else len(block)
            sections[key] = block[start:end].strip()

        sections["FINDINGS"] = ""

    return sections



def eval_single_subject(args):
    metadata_json_pth=os.path.join(args.subject_folder, 'patient_metadata_btreport.json')
    metadata = load_json(metadata_json_pth)
    real_reports = parse_radiology_report(metadata[args.real_report_key])
    synthetic_reports = parse_radiology_report(metadata[args.synthetic_report_key])

    keys=['BRAIN', 'MASS EFFECT & VENTRICLES',]
    refs = ["\n\n".join(f"{k}:\n{real_reports[k]}" for k in keys if k in real_reports)]
    hyps = ["\n\n".join(f"{k}:\n{synthetic_reports[k]}" for k in keys if k in synthetic_reports)]


    print(refs)
    print('--'*30)
    print('-*-*'*30)
    print(hyps)

    evaluator = RadEval(
        do_rouge=True,
        do_bertscore=True,
        do_srr_bert=True,
        do_ratescore=True,
        do_details=args.do_details
    )
    
    results = evaluator(refs=refs, hyps=hyps)
    print(json.dumps(results, indent=2))

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
    parser.add_argument("--subject_folder", type=str, help="Path to the subject folder containing the MRI data.", required=True)
    parser.add_argument("--real_report_key", type=str, default='Clinical Report')
    parser.add_argument("--synthetic_report_key", type=str, default='llm report')
    parser.add_argument("--devices", type=str, default='0', help="String with cuda device IDs for use by synthseg and SynthMorph. E.g. '0,1' or '0'.")
    parser.add_argument("--do_details", action="store_true")


    # parser.add_argument("--et_label", type=int, default=4)
    # parser.add_argument("--llm", type=str, default="gpt-oss:120b")


    args = parser.parse_args()

    subject = os.path.basename(os.path.normpath(args.subject_folder))
    logger = get_logger(subject) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    eval_single_subject(args)
