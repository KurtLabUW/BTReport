import os
import json, csv
import argparse
from pathlib import Path
from .eval import eval_single_subject  
from .utils.log import get_logger

def eval_all(args):
    root = Path(args.root_folder)
    assert root.exists(), f"Folder not found: {root}"

    subjects = [p for p in root.iterdir() if (p / "patient_metadata_btreport.json").exists()]

    print(f"\n Found {len(subjects)} subjects for evaluation in:\n{root}\n")

    aggregate_results = {}
    log_results = {}

    for i, subject_folder in enumerate(subjects, 1):
        if 'BraTS' in str(subject_folder):
            continue

        result_file = subject_folder / "eval_results.json"
        if args.do_details:
            result_file = result_file.with_name(result_file.stem + "_details.json")
        if result_file.exists() and args.skip_processed:
            continue

        print(f"[{i}/{len(subjects)}] -- Evaluating: {subject_folder.name}")

        sub_args = argparse.Namespace(
            subject_folder=str(subject_folder),
            real_report_key=args.real_report_key,
            synthetic_report_key=args.synthetic_report_key,
            devices=args.devices,
            do_details=args.do_details,
            llm=args.llm,
            parse_real=args.parse_real,
            parse_synthetic=args.parse_synthetic,
        )

        success, message = eval_single_subject(sub_args)
        log_results[subject_folder.name] = {'success':success, 'message': message}

        # Load results saved by eval_single_subject
        if result_file.exists():
            with open(result_file, "r") as f:
                aggregate_results[subject_folder.name] = json.load(f)

    # log successful/unsuccessful runs
    log_path = root / "eval_logs.csv"
    if args.do_details:
        log_path = log_path.with_name(log_path.stem + "_details.json")
    with open(log_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "success", "message"])   # header
        for subject, info in log_results.items():
            writer.writerow([subject, info["success"], info["message"]])

    # Save aggregated results
    save_path = root / "eval_summary_all.json"
    if args.do_details:
        save_path = save_path.with_name(save_path.stem + "_details.json")
        
    with open(save_path, "w") as f:
        json.dump(aggregate_results, f, indent=2)

    print(f"\n Summary saved to {save_path}")
    print(f"Done evaluating {len(subjects)} subjects.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation across all subjects.")
    parser.add_argument("--root_folder", type=str, required=True, help="Folder containing subject subfolders")
    parser.add_argument("--real_report_key", type=str, default='Clinical Report')
    parser.add_argument("--synthetic_report_key", type=str, default='BTReport Generated Report (gpt-oss:120b)')
    parser.add_argument("--devices", type=str, default='0,1,2,3')
    parser.add_argument("--do_details", action="store_true")
    parser.add_argument("--llm", type=str, default="deepseek-r1:70b")
    parser.add_argument("--skip_processed", action="store_true")

    parser.add_argument("--parse-real", action=argparse.BooleanOptionalAction, default=True, help="Parse the real clinical reports, use --no-parse-real for False",)
    parser.add_argument("--parse-synthetic", action=argparse.BooleanOptionalAction, default=True, help="Parse the synthetic generated reports, use --no-parse-synthetic for False",)


    args = parser.parse_args()


    logger = get_logger(args.root_folder) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


    eval_all(args)
f