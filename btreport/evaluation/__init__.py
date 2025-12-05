
import os, json

def merge_reports_into_json(root_dir,
                            real_report_key="Clinical Report",
                            generated_report_key="BTReport Generated Report (llama3:70b)",
                            output_filename="merged_reports_btreport_llama3_70b.json"):
    """
    Walks through all subfolders of root_dir.
    Reads patient_metadata_btreport.json in each folder.
    Extracts real_report_key and generated_report_key.
    Saves a merged JSON file at: root_dir/output_filename
    """

    merged = {}

    # iterate through all subfolders
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        meta_path = os.path.join(folder_path, "patient_metadata_btreport.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Failed to read {meta_path}: {e}")
            continue

        real_report = meta.get(real_report_key, "")
        predicted_report = meta.get(generated_report_key, "")

        merged[folder] = {
            "Clinical Report": real_report,
            "Predicted Report": predicted_report
        }

    # Save to JSON inside the root_dir
    output_path = os.path.join(root_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved merged JSON to: {output_path}")
    return output_path


if __name__ == '__main__':
     merge_reports_into_json(root_dir='../../data/example')