#!/usr/bin/env python3
import csv, sys, os, math, subprocess, datetime

def run_apply(transform, moving, moved, wrapper, subjects_dir):
    """Run SynthMorph apply for one image or segmentation."""
    is_seg = "seg" in os.path.basename(moving).lower() or "label" in os.path.basename(moving).lower()
    cmd = [wrapper, "apply"]
    if is_seg:
        cmd += ["-m", "nearest"]
    cmd += [transform, moving, moved]
    subprocess.run(cmd, check=True)

def main():
    if len(sys.argv) != 4:
        print("Usage: apply_split.py <paths.csv> <split_idx> <num_splits>")
        sys.exit(1)

    csv_path, split_idx, num_splits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    wrapper = "/gscratch/kurtlab/synthmorph/synthmorph"
    subjects_dir = os.environ.get("SUBJECTS_DIR", "/gscratch/kurtlab")

    log_dir = os.path.join(os.path.dirname(csv_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_csv = os.path.join(log_dir, f"apply_split_{split_idx:03d}_log.csv")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    chunk_size = math.ceil(total / num_splits)
    start = split_idx * chunk_size
    end = min(start + chunk_size, total)
    if start >= total:
        return

    fieldnames = ["row", "moving", "status", "time"]
    write_header = not os.path.exists(log_csv)

    with open(log_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(rows[start:end], start=start):
            transform, moving, moved = (row[k].strip() for k in row)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            moved_abs = os.path.join(subjects_dir, moved)
            transform_abs = os.path.join(subjects_dir, transform)
            moving_abs = os.path.join(subjects_dir, moving)

            if not all([transform, moving, moved]):
                writer.writerow({"row": i, "moving": moving, "status": "missing_field", "time": timestamp})
                f.flush()
                continue

            if not (os.path.exists(moving_abs) and os.path.exists(transform_abs)):
                writer.writerow({"row": i, "moving": moving, "status": "missing_inputs", "time": timestamp})
                f.flush()
                continue

            if os.path.exists(moved_abs) and os.path.getsize(moved_abs) > 0:
                writer.writerow({"row": i, "moving": moving, "status": "already_done", "time": timestamp})
                f.flush()
                continue

            os.makedirs(os.path.dirname(moved_abs), exist_ok=True)

            try:
                run_apply(transform, moving, moved, wrapper, subjects_dir)
                writer.writerow({"row": i, "moving": moving, "status": "success", "time": timestamp})
            except subprocess.CalledProcessError:
                writer.writerow({"row": i, "moving": moving, "status": "failed", "time": timestamp})

            f.flush()

if __name__ == "__main__":
    main()
