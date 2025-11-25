from .utils import register, plotting, anat_segmentation
from .llm_report_generation import ollama_report_gen
from .midline_shift.midline_shift3d import midline_shift_3d
from .vasari_features.extract_vasari_features import vasari_features
import os, shutil, glob, json
import argparse, logging
from os.path import join as join

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


"""
export SYNTHMORPH_SIF=/pscratch/sd/j/jehr/synthmorph/synthmorph_4.sif
export PATH=${PATH}:/cvmfs/oasis.opensciencegrid.org/mis/apptainer/1.3.3/x86_64/bin
export SYNTHSEG_SIF=/pscratch/sd/j/jehr/synthseg/synthseg.sif
"""


def get_modality_paths(d):
    return {
        "t1": glob.glob(os.path.join(d, "*-t1.nii.gz"))[0],
        "t2": glob.glob(os.path.join(d, "*-t2.nii.gz"))[0],
        "t2f": glob.glob(os.path.join(d, "*-t2f.nii.gz"))[0],
        "t1c": glob.glob(os.path.join(d, "*-t1c.nii.gz"))[0],
    }


def register_atlas(subject_folder: str):
    print(subject_folder)
    pass


def main(args: argparse.Namespace):
    # modality_paths = get_modality_paths(args.subject_folder)
    # t1_path = modality_paths['t1']
    t1_path = glob.glob(os.path.join(args.subject_folder, "*-t1.nii.gz"))[0]
    tumor_path = glob.glob(os.path.join(args.subject_folder, "*-seg.nii.gz"))[0]

    tmp_dir = join(args.subject_folder, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Register atlas to image, image to atlas, and midline
    mni_in_subj = join(tmp_dir, "MNI152_in_subject_space.nii.gz")
    mni_tfm = join(tmp_dir, "MNI152_in_subject_space_transform.nii.gz")

    sub_in_mni = join(tmp_dir, "subject_in_MNI152_space.nii.gz")
    sub_tfm = join(tmp_dir, "subject_in_MNI152_space_transform.nii.gz")

    tum_in_mni = join(tmp_dir, "tumor_seg_in_MNI152_space.nii.gz")

    midline_out = join(tmp_dir, "patient_midline.nii.gz")

    logger.info(f'Starting registration steps...')
    register.register_mni_to_subject(
        fixed=t1_path, moved=mni_in_subj, transform=mni_tfm, overwrite=args.overwrite
    )  # register MNI152 to subject space
    register.register_to_mni(moving=t1_path, moved=sub_in_mni, transform=sub_tfm, overwrite=args.overwrite)  # register T1 to MNI152 space
    register.register_midline_to_subject(
        moved=midline_out, transform=mni_tfm, overwrite=args.overwrite
    )  # register MNI152 midline to subject space using mni_tfm
    register.apply_transform(moving=tumor_path, moved=tum_in_mni, transform=sub_tfm, is_seg=True)
    logger.info(f'Finished registration steps!')


    # SynthSeg is unreliable on images with tumors, so we run it on the (healthy) MNI atlas registered to the subject space
    logger.info(f'Starting anatomical segmentation steps...')
    anatseg = mni_in_subj.replace(".nii.gz", "_synthseg.nii.gz")
    merged_seg = mni_in_subj.replace(".nii.gz", "_merged_seg.nii.gz")
    anat_segmentation.synthseg(input_path=mni_in_subj, output_path=anatseg)

    # Merge tumor, midline, and anatomical segmentation masks
    anat_segmentation.merge_tumor_midline_and_anat_masks(
        synthseg_path=anatseg,
        tumor_path=tumor_path,
        midline_path=midline_out,
        save_path=merged_seg,
        ncr_label=args.ncr_label,
        ed_label=args.ed_label,
        et_label=args.et_label,
    )
    logger.info(f'Finished segmentation steps! Merged mask can be found in {merged_seg}')

    # Extract metadata to be used in the report
    metadata = {}  # TODO: load from json with load_metadata, make sure it includes tumor_type for merged, add tumor measurements

    midline_summary = midline_shift_3d(tmp_dir=tmp_dir, tumor=tumor_path, ncr_label=args.ncr_label, ed_label=args.ed_label, et_label=args.et_label, overwrite=args.overwrite)
    metadata.update(midline_summary)

    vasari_summary = vasari_features(tumor=tumor_path, tumor_mni=tum_in_mni, metadata=metadata, merged=merged_seg, verbose=False, translate=True, ncr_label=args.ncr_label, ed_label=args.ed_label, et_label=args.et_label)
    metadata.update(vasari_summary)

    with open(join(args.subject_folder, "patient_metadata_btreport.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # print(metadata)

    if args.clear_tmp:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a brain tumor report for one subject.")
    parser.add_argument("--subject_folder", type=str, help="Path to the subject folder containing the MRI data.")

    parser.add_argument(
        "--output",
        type=str,
        default="report.json",
        help="Path to save the generated report (default: report.json).",
    )

    parser.add_argument("--clear_tmp", action="store_true", help="Delete the temporary directory after processing.")
    parser.add_argument("--overwrite", action="store_true", help="Redo this step, overwriting previous results.")
    parser.add_argument("--ncr_label", type=int, default=1)
    parser.add_argument("--ed_label", type=int, default=2)
    parser.add_argument("--et_label", type=int, default=4)

    args = parser.parse_args()

    main(args)
