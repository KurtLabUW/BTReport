import csv, sys, os, math, subprocess, datetime

def run_registration(moving, fixed, moved, transform, wrapper, subjects_dir):
    """Run SynthMorph registration for one subject."""
    print(f"\n========================================")
    print(f"    SynthMorph Registration")
    print(f"----------------------------------------")
    print(f"  Moving    : {subjects_dir}/{moving}")
    print(f"  Fixed     : {subjects_dir}/{fixed}")
    print(f"  Output    : {subjects_dir}/{moved}")
    print(f"  Transform : {subjects_dir}/{transform}")
    print(f"========================================")

    cmd = [wrapper, "register", "-g",
           "-o", moved, "-t", transform,
           moving, fixed]
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) != 4:
        print("Usage: register_split.py <paths.csv> <split_idx> <num_splits>")
        sys.exit(1)

    csv_path, split_idx, num_splits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    wrapper = "/gscratch/kurtlab/synthmorph/synthmorph"
    subjects_dir = os.environ.get("SUBJECTS_DIR", "/gscratch/kurtlab")

    log_dir = os.path.join(os.path.dirname(csv_path), "logs")
    os.makedirs(log_dir, exist_ok=True)

    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    log_csv = os.path.join(log_dir, f"{csv_name}_split_{split_idx:03d}_log.csv")

    # Load all CSV rows (skip header)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    chunk_size = math.ceil(total / num_splits)
    start = split_idx * chunk_size
    end = min(start + chunk_size, total)

    if start >= total:
        print(f"Split {split_idx} has no data (total rows={total}). Exiting.")
        return

    print(f"Processing split {split_idx+1}/{num_splits}: rows {start+1}-{end} of {total}")

    with open(log_csv, "w", newline="") as log_f:
        fieldnames = ["row", "moving", "status", "time"]
        writer = csv.DictWriter(log_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows[start:end], start=start):
            moving, fixed, moved, transform = (row[k].strip() for k in row)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not all([moving, fixed, moved, transform]):
                print(f"Skipping row {i}: missing field(s)")
                writer.writerow({"row": i, "moving": moving, "status": "missing", "time": timestamp})
                log_f.flush()
                continue

            moved_abs = os.path.join(subjects_dir, moved)
            transform_abs = os.path.join(subjects_dir, transform)

            # Skip if output already exists
            if os.path.exists(moved_abs) and os.path.getsize(moved_abs) > 0:
                print(f"Skipping row {i}: already processed ({moved})")
                writer.writerow({"row": i, "moving": moving, "status": "already_done", "time": timestamp})
                log_f.flush()
                continue

            # Ensure output dirs exist
            os.makedirs(os.path.dirname(moved_abs), exist_ok=True)
            os.makedirs(os.path.dirname(transform_abs), exist_ok=True)

            try:
                run_registration(moving, fixed, moved, transform, wrapper, subjects_dir)
                writer.writerow({"row": i, "moving": moving, "status": "success", "time": timestamp})
            except subprocess.CalledProcessError:
                print(f"Registration failed for row {i}")
                writer.writerow({"row": i, "moving": moving, "status": "failed", "time": timestamp})

            # flush to ensure regular updates
            log_f.flush()

    print(f"\n📝 Log written to {log_csv}")


if __name__ == "__main__":
    main()




# import csv, sys, os, math, subprocess, datetime

# def run_registration(moving, fixed, moved, transform, wrapper, subjects_dir):
#     """Run SynthMorph registration for one subject."""
#     print(f"\n========================================")
#     print(f"    SynthMorph Registration")
#     print(f"----------------------------------------")
#     print(f"  Moving    : {subjects_dir}/{moving}")
#     print(f"  Fixed     : {subjects_dir}/{fixed}")
#     print(f"  Output    : {subjects_dir}/{moved}")
#     print(f"  Transform : {subjects_dir}/{transform}")
#     print(f"========================================")

#     cmd = [wrapper, "register", "-g",
#            "-o", moved, "-t", transform,
#            moving, fixed]
#     subprocess.run(cmd, check=True)


# def main():
#     if len(sys.argv) != 4:
#         print("Usage: register_split.py <paths.csv> <split_idx> <num_splits>")
#         sys.exit(1)

#     csv_path, split_idx, num_splits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
#     wrapper = "/gscratch/kurtlab/synthmorph/synthmorph"
#     subjects_dir = os.environ.get("SUBJECTS_DIR", "/gscratch/kurtlab")

#     log_dir = os.path.join(os.path.dirname(csv_path), "logs")
#     os.makedirs(log_dir, exist_ok=True)
#     log_csv = os.path.join(log_dir, f"split_{split_idx:03d}_log.csv")

#     # Load all CSV rows (skip header)
#     with open(csv_path) as f:
#         reader = csv.DictReader(f)
#         rows = list(reader)

#     total = len(rows)
#     chunk_size = math.ceil(total / num_splits)
#     start = split_idx * chunk_size
#     end = min(start + chunk_size, total)

#     if start >= total:
#         print(f"Split {split_idx} has no data (total rows={total}). Exiting.")
#         return

#     print(f"Processing split {split_idx+1}/{num_splits}: rows {start+1}–{end} of {total}")

#     results = []  # rows for output log

#     for i, row in enumerate(rows[start:end], start=start):
#         moving, fixed, moved, transform = (row[k].strip() for k in row)
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         if not all([moving, fixed, moved, transform]):
#             print(f"Skipping row {i}: missing field(s)")
#             results.append({"row": i, "moving": moving, "status": "missing", "time": timestamp})
#             continue

#         moved_abs = os.path.join(subjects_dir, moved)
#         transform_abs = os.path.join(subjects_dir, transform)

#         # Skip if output already exists
#         if os.path.exists(moved_abs) and os.path.getsize(moved_abs) > 0:
#             print(f"Skipping row {i}: already processed ({moved})")
#             results.append({"row": i, "moving": moving, "status": "already_done", "time": timestamp})
#             continue

#         # Ensure output dirs exist
#         os.makedirs(os.path.dirname(moved_abs), exist_ok=True)
#         os.makedirs(os.path.dirname(transform_abs), exist_ok=True)

#         try:
#             run_registration(moving, fixed, moved, transform, wrapper, subjects_dir)
#             results.append({"row": i, "moving": moving, "status": "success", "time": timestamp})
#         except subprocess.CalledProcessError:
#             print(f"Registration failed for row {i}")
#             results.append({"row": i, "moving": moving, "status": "failed", "time": timestamp})

#     # Write log CSV for this split
#     with open(log_csv, "w", newline="") as f:
#         fieldnames = ["row", "moving", "status", "time"]
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)

#     print(f"\n📝 Log written to {log_csv}")


# if __name__ == "__main__":
#     main()
