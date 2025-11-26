import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(
        description="Run generate_report.py on ALL subject folders inside a directory."
    )

    parser.add_argument("--root_folder", type=str, required=True,
                        help="Path containing many subject subfolders.")

    parser.add_argument("--clear_tmp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ncr_label", type=int, default=1)
    parser.add_argument("--ed_label", type=int, default=2)
    parser.add_argument("--et_label", type=int, default=4)
    parser.add_argument("--llm", type=str, default="gpt-oss:120b")

    args = parser.parse_args()

    root = args.root_folder

    generate_script = "btreport.generate_report"

    for entry in sorted(os.listdir(root)):
        subject_dir = os.path.join(root, entry)

        if not os.path.isdir(subject_dir):
            continue  # skip files

        print(f"\n=== Processing {subject_dir} ===")

        cmd = [
            "python3",
            "-m",
            generate_script,
            "--subject_folder", subject_dir,
            "--ncr_label", str(args.ncr_label),
            "--ed_label", str(args.ed_label),
            "--et_label", str(args.et_label),
            "--llm", args.llm,
        ]

        if args.clear_tmp:
            cmd.append("--clear_tmp")
        if args.overwrite:
            cmd.append("--overwrite")

        subprocess.run(cmd, check=True)

    print("\nFinished processing all subjects.")


if __name__ == "__main__":
    main()
