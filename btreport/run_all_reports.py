import argparse
import subprocess
import os
import json

import csv
import time
import socket
import fcntl
from datetime import datetime


from .utils.log import get_logger

# logger = get_logger("btreport.run_all_reports", subject="run_all")
logger = get_logger("batchgen")


# def main():
#     parser = argparse.ArgumentParser(description="Run generate_report.py on ALL subject folders inside a directory.")

#     parser.add_argument("--root_folder", type=str, required=True, help="Path containing many subject subfolders.")

#     parser.add_argument("--clear_tmp", action="store_true")
#     parser.add_argument("--overwrite", action="store_true")
#     parser.add_argument("--ncr_label", type=int, default=1)
#     parser.add_argument("--ed_label", type=int, default=2)
#     parser.add_argument("--et_label", type=int, default=4)
#     parser.add_argument("--llm", type=str, default="gpt-oss:120b")
#     parser.add_argument("--eval", action="store_true", help="Running evaluation after generation.")

#     args = parser.parse_args()

#     root = os.path.realpath(args.root_folder)
#     llm = args.llm

#     generate_script = "btreport.generate_report"

#     for entry in sorted(os.listdir(root)):
#         subject_dir = os.path.join(root, entry)

#         if not os.path.isdir(subject_dir):
#             continue  # skip files

#         logger.info(f"\n=== Processing {subject_dir} ===")

#         cmd = [
#             "python3",
#             "-m",
#             generate_script,
#             "--subject_folder",
#             subject_dir,
#             "--ncr_label",
#             str(args.ncr_label),
#             "--ed_label",
#             str(args.ed_label),
#             "--et_label",
#             str(args.et_label),
#             "--llm",
#             llm,
#         ]

#         if args.clear_tmp:
#             cmd.append("--clear_tmp")
#         if args.overwrite:
#             cmd.append("--overwrite")

#         subprocess.run(cmd, check=True)

#         update_merged_reports(
#             root_dir=root,
#             subject_id=entry,
#             llm=llm,
#         )

#     logger.info("\nFinished processing all subjects.")

def main():
    parser = argparse.ArgumentParser(
        description="Run generate_report.py on subject folders with optional array-job splitting."
    )

    parser.add_argument("--root_folder", type=str, required=True)

    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--split_no", type=int, default=0)

    parser.add_argument("--clear_tmp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ncr_label", type=int, default=1)
    parser.add_argument("--ed_label", type=int, default=2)
    parser.add_argument("--et_label", type=int, default=4)
    parser.add_argument("--llm", type=str, default="gpt-oss:120b")
    parser.add_argument("--run_name", type=str, default="v0")

    parser.add_argument("--eval", action="store_true")
    
    # parser.add_argument("--merged_json", type=str, default="merged_reports_btreport.json")



    args = parser.parse_args()


    if not (0 <= args.split_no < args.num_splits):
        raise ValueError("--split_no must satisfy 0 <= split_no < num_splits")

    root = os.path.realpath(args.root_folder)
    args.merged_json = os.path.join(root, f"merged_reports_btreport_{args.llm}.json")

    llm = args.llm
    generate_script = "btreport.generate_report"

    # ----------------------------------
    # Runtime CSV setup
    # ----------------------------------
    runtime_csv = os.path.join(root, "runtime_per_subject.csv")
    csv_header = [
        "subject_id",
        "split_no",
        "hostname",
        "start_time",
        "end_time",
        "runtime_seconds",
    ]
    hostname = socket.gethostname()

    all_subjects = sorted(
        entry for entry in os.listdir(root)
        if os.path.isdir(os.path.join(root, entry))
    )

    split_subjects = all_subjects[args.split_no :: args.num_splits]

    logger.info(
        f"Total subjects: {len(all_subjects)} | "
        f"Split {args.split_no}/{args.num_splits} -> "
        f"{len(split_subjects)} subjects"
    )


    for ix, entry in enumerate(split_subjects):
        subject_dir = os.path.join(root, entry)
        logger.info(f"\n[{ix+1}/{len(split_subjects)}]\n=== Processing {entry} ===")

        start_ts = time.time()
        start_iso = datetime.utcnow().isoformat()

        cmd = [
            "python3",
            "-m",
            generate_script,
            "--subject_folder", subject_dir,
            "--ncr_label", str(args.ncr_label),
            "--ed_label", str(args.ed_label),
            "--et_label", str(args.et_label),
            "--llm", llm,
            "--run_name", args.run_name,
        ]

        if args.clear_tmp:
            cmd.append("--clear_tmp")
        if args.overwrite:
            cmd.append("--overwrite")

        try:
            subprocess.run(cmd, check=True)

            update_merged_reports(
                root_dir=root,
                subject_id=entry,
                llm=llm,
                run_name=args.run_name,
                output_filename= args.merged_json,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Generate failed for {entry}")
            logger.error(e)
            continue  

        end_ts = time.time()
        end_iso = datetime.utcnow().isoformat()
        runtime = end_ts - start_ts

        append_runtime_csv(
            runtime_csv,
            row={
                "subject_id": entry,
                "split_no": args.split_no,
                "hostname": hostname,
                "start_time": start_iso,
                "end_time": end_iso,
                "runtime_seconds": f"{runtime:.2f}",
            },
            header=csv_header,
        )

        logger.info(
            f"Finished {entry} in {runtime:.2f} seconds"
        )

    logger.info(f"\nFinished processing split {args.split_no}.")




def append_runtime_csv(csv_path, row, header):
    """
    Append a row to a CSV using an exclusive file lock.
    Creates file + header if it does not exist.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)

        write_header = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=header)

        if write_header:
            writer.writeheader()

        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def update_merged_reports(root_dir, subject_id, llm, run_name, real_report_key="Clinical Report", output_filename="merged_reports_btreport.json", subject_report_filename="patient_metadata_btreport.json"):
    merged_path = os.path.join(root_dir, output_filename)
    if os.path.exists(merged_path):
        with open(merged_path, "r") as f:
            merged = json.load(f)
    else:
        merged = {}

    subject_report_path = os.path.join(root_dir, subject_id, subject_report_filename)
    if not os.path.exists(subject_report_path):
        logger.info(f"Missing {subject_report_path}, skipping merge.")
        return

    try:
        with open(subject_report_path, "r") as f:
            subject_btreport = json.load(f)
    except Exception as e:
        logger.info(f"Failed to read {subject_report_path}: {e}")
        return

    bt_generated_key = f"BTReport Generated Report ({llm}, run_name={run_name})"
    predicted_key = f"Predicted Report ({llm}, run_name={run_name} )"

    if subject_id not in merged:
        merged[subject_id] = {}

    # Store prediction
    if bt_generated_key not in subject_btreport:
        logger.info(f"Merge error: '{bt_generated_key}' missing in {subject_report_path}")
        return
    else:
        merged[subject_id][predicted_key] = subject_btreport[bt_generated_key]


    # Store ground truth if it exists
    if real_report_key not in subject_btreport:
        logger.info(f"Merge warning: '{real_report_key}' missing in {subject_report_path}")
    else:
        merged[subject_id][real_report_key] = subject_btreport[real_report_key]



    # Safe write
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)

    with open(merged_path, "a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        # Load latest on-disk content
        lock_f.seek(0)
        try:
            existing = json.load(lock_f)
        except json.JSONDecodeError:
            existing = {}

        # Merge in new content
        existing.update(merged)

        # Write atomically via temp file
        tmp_path = merged_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(existing, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        if not os.path.exists(tmp_path):
            logger.warning(
                f"Temp file {tmp_path} missing; skipping replace."
            )
            fcntl.flock(lock_f, fcntl.LOCK_UN)
            return

        os.replace(tmp_path, merged_path)


        fcntl.flock(lock_f, fcntl.LOCK_UN)




if __name__ == "__main__":
    main()
