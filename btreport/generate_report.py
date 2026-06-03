from .utils import register, plotting, anat_segmentation
from .utils.log import get_logger
from .llm_report_generation.factory import generate_llm_report, resolve_backend
from .midline_shift.midline_shift3d import midline_shift_3d
from .vasari_features import ExtractVASARI

# from .vasari_features.extract_vasari_features import vasari_features

import os, shutil, glob, json
import argparse
from os.path import join



def main(args: argparse.Namespace):
    t1_paths = glob.glob(os.path.join(args.subject_folder, "*-t1n.nii.gz"))
    t1c_paths = glob.glob(os.path.join(args.subject_folder, "*-t1c.nii.gz"))
    if t1_paths:
        t1_path = t1_paths[0]
    elif t1c_paths:
        t1_path = t1c_paths[0]
    else:
        raise FileNotFoundError(f"No T1 or T1c image found in {args.subject_folder}")
    tumor_path = glob.glob(os.path.join(args.subject_folder, "*-seg.nii.gz"))[0]

    tmp_dir = join(args.subject_folder, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Load patient metadata from metadata.json in subject folder
    metadata_json_pth = join(args.subject_folder, "metadata.json")
    if not os.path.exists(metadata_json_pth):
        metadata = {}
    else:
        with open(metadata_json_pth, "r") as f:
            metadata = json.load(f)

    # Load in previous report if it exists
    report_save_path = join(args.subject_folder, "patient_metadata_btreport.json")
    if os.path.exists(report_save_path):
        with open(report_save_path, "r") as f:
            existing_report = json.load(f)
        logger.info(f"Found previously generated metadata, loading this..")
        metadata = {**existing_report, **metadata}

    # Register atlas to image, image to atlas, and midline
    mni_in_subj = join(tmp_dir, "MNI152_in_subject_space.nii.gz")
    mni_tfm = join(tmp_dir, "MNI152_in_subject_space_transform.nii.gz")

    sub_in_mni = join(tmp_dir, "subject_in_MNI152_space.nii.gz")
    sub_tfm = join(tmp_dir, "subject_in_MNI152_space_transform.nii.gz")

    tum_in_mni = join(tmp_dir, "tumor_seg_in_MNI152_space.nii.gz")

    patient_midline = join(tmp_dir, "patient_midline.nii.gz")
    ideal_midline = join(tmp_dir, "ideal_midline.nii.gz")
    midline_distances = join(tmp_dir, "midline_distances.nii.gz")

    logger.info(f"** [0/4] Starting registration steps...")
    register.register_mni_to_subject(fixed=t1_path, moved=mni_in_subj, transform=mni_tfm, overwrite=args.overwrite)  # register MNI152 to subject space
    register.register_to_mni(moving=t1_path, moved=sub_in_mni, transform=sub_tfm, overwrite=args.overwrite)  # register T1 to MNI152 space
    register.register_midline_to_subject(moved=patient_midline, transform=mni_tfm, overwrite=args.overwrite)  # register MNI152 midline to subject space using mni_tfm
    register.apply_transform(moving=tumor_path, moved=tum_in_mni, transform=sub_tfm, is_seg=True)  # register tumor mask to MNI152 space using sub_tfm
    logger.info(f"* Finished registration steps!")

    # SynthSeg is unreliable on images with tumors, so we run it on the (healthy) MNI atlas registered to the subject space, then overlay the tumor mask.
    logger.info(f"** [1/4] Starting anatomical segmentation steps...")
    anatseg = mni_in_subj.replace(".nii.gz", "_synthseg.nii.gz")
    merged_seg = mni_in_subj.replace(".nii.gz", "_merged_seg.nii.gz")
    anat_segmentation.synthseg(input_path=mni_in_subj, output_path=anatseg)

    # Merge tumor, midline, and anatomical segmentation masks
    overlap_regions = anat_segmentation.merge_tumor_midline_and_anat_masks(
        synthseg_path=anatseg,
        tumor_path=tumor_path,
        midline_path=patient_midline,
        save_path=merged_seg,
        ncr_label=args.ncr_label,
        ed_label=args.ed_label,
        et_label=args.et_label,
        tumor_type=metadata.get("tumor-type", "glioma"),
        overwrite=args.overwrite,
    )
    metadata.update({"Anatomical Overlap Regions": overlap_regions})

    logger.info(f"* Finished segmentation steps! Merged mask can be found in {merged_seg}")

    # Extract midline shift features
    logger.info(f"** [2/4] Starting midline shift processing...")
    midline_summary = midline_shift_3d(tumor=tumor_path, 
                                        deformed_midline_path=patient_midline,
                                        ideal_midline_path=ideal_midline,
                                        midline_distances_path=midline_distances,
                                        anat_seg_path=anatseg,
                                        ncr_label=args.ncr_label, ed_label=args.ed_label, et_label=args.et_label, overwrite=args.overwrite)
    metadata.update(midline_summary)

    # Extract VASARI features
    # vasari_summary = vasari_features(tumor=tumor_path, tumor_mni=tum_in_mni, metadata=metadata, merged=merged_seg, verbose=False, ncr_label=args.ncr_label, ed_label=args.ed_label, et_label=args.et_label)
    logger.info(f"** [3/4] Starting VASARI feature extraction steps...")
    extractor = ExtractVASARI(enhancing_label=args.et_label, nonenhancing_label=args.ncr_label, oedema_label=args.ed_label, verbose=False)
    vasari_summary = extractor(tumorseg_mni=tum_in_mni, tumorseg_ss=tumor_path, merged=merged_seg, metadata=metadata)
    metadata.update(vasari_summary)

    logger.info(
        f"** [4/4] Starting report generation with LLM ({args.llm}, backend={resolve_backend(args.llm)})..."
    )
    metadata_no_clinical = {k: v for k, v in metadata.items() if k != "Clinical Report"}

    keys_to_keep = [
        "Anatomical Overlap Regions",
        "Tumor Location",
        "Side of Tumor Epicenter",
        "Number of lesions",
        "Multifocal or Multicentric",
        "Multiple satellites present",
        "Cortical involvement",
        "Deep WM invasion",
        "Ependymal (ventricular) Invasion",
        # "Eloquent Brain Involvement",
        "Enlarged Ventricles",
        "Asymmetrical Ventricles",
        "Edema crosses midline",
        "CET Crosses midline",
        "Enhancement Quality",
        "Thickness of enhancing margin",
        # "NCR Volume (mL)",
        # "ED Volume (mL)",
        # "ET Volume (mL)",
        # "Total tumor volume (mL)",
        "Proportion Enhancing",
        "Proportion Necrosis",
        "Proportion of Oedema",
        "Effaced Ventricle",
        "Lesion Sizes APxTVxCC (cm)",
        # "Region Proportions",
        # "max_shift_mm",
        # "level_max_shift",
        "midline_shift_present",
        "Text Report",
    ]
    if metadata_no_clinical['midline_shift_present'] == "Yes":
        keys_to_keep+=["level_max_shift", "max_shift_mm"]

    refined_metadata = {k: v for k, v in metadata_no_clinical.items() if k in keys_to_keep}
    
    json_save_key = f"BTReport Generated Report ({args.llm}, run_name={args.run_name})"
    if json_save_key not in metadata:
        args.image_path = join(args.subject_folder, "tumor_maxslice.png") if args.image else None
        report = generate_llm_report(args.subject_folder.split("/")[-1], refined_metadata, model=args.llm, image_path=args.image_path)
        logger.info(f"* Finished LLM report generation using extracted metadata!")
        metadata[json_save_key] = report
    else:
        logger.info(f'Key {json_save_key} found in metadata, skipping LLM report')

    with open(report_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f'Saved extracted metadata and LLM report to {join(args.subject_folder, "patient_metadata_btreport.json")} as {json_save_key}')

    if args.clear_tmp:  # Delete intermediate files after processing, useful for memory reduction but you lose interpretability of results.
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a brain tumor report for one subject.")
    parser.add_argument("--subject_folder", type=str, help="Path to the subject folder containing the MRI data.")

    parser.add_argument("--clear_tmp", action="store_true", help="Delete the temporary directory after processing.")
    parser.add_argument("--overwrite", action="store_true", help="Redo this step, overwriting previous results.")
    parser.add_argument("--ncr_label", type=int, default=1)
    parser.add_argument("--ed_label", type=int, default=2)
    parser.add_argument("--et_label", type=int, default=3)
    parser.add_argument("--devices", type=str, default="0", help="String with cuda device IDs for use by synthseg and SynthMorph. E.g. '0,1' or '0'.")
    parser.add_argument("--run_name", type=str, default='v0')

    parser.add_argument(
        "--image",
        action="store_true",
        help="Indicator as to whther the model will use images for generation. Will look for tumor_maxslice.png in subject_folder",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-5.4-mini",
        help=(
            "Model name; backend is inferred (OpenAI: gpt-5.4-mini; Ollama: llama3:70b). "
            "Override with openai-<model> or ollama-<model>."
        ),
    )

    args = parser.parse_args()

    subject = os.path.basename(os.path.normpath(args.subject_folder))
    logger = get_logger(subject)
    # logger = get_logger("btreport.subject", subject=subject)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    main(args)
