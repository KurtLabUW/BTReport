from . import register
from .log import get_logger

import argparse, glob, os, shutil, json
from os.path import join
from tqdm import tqdm


CISTERN = 'btreport/utils/Cistern_Segmentations.nii.gz'

def main(args):

    if args.subject_folder:
        all_subjects = [args.subject_folder]
    else:
        all_subjects = [join(args.root_folder, f) for f in os.listdir(args.root_folder) if os.path.isdir(join(args.root_folder, f))]

    for subject_folder in tqdm(all_subjects, total=len(all_subjects), colour='red'):
        tmp_dir = join(subject_folder, "tmp")

        mni_tfm = join(tmp_dir, "MNI152_in_subject_space_transform.nii.gz")

        cistern_in_subj = join(tmp_dir, "cistern_in_subject_space.nii.gz")


        register.apply_transform(moving=CISTERN, moved=cistern_in_subj, transform=mni_tfm, is_seg=True)
        logger.info(f'* Finished registration steps!')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a brain tumor report for one subject.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject_folder", type=str, help="Path to a single subject folder.")
    group.add_argument("--root_folder", type=str, help="Path to a root directory containing many subject folders.")

    # parser.add_argument("--ncr_label", type=int, default=1)
    # parser.add_argument("--ed_label", type=int, default=2)
    # parser.add_argument("--et_label", type=int, default=4)
    parser.add_argument("--devices", type=str, default='0', help="String with cuda device IDs for use by synthseg and SynthMorph. E.g. '0,1' or '0'.")

    # parser.add_argument("--image_path", type=str, default=None, help="String with cuda device IDs for use by synthseg and SynthMorph. E.g. '0,1' or '0'.")
    # parser.add_argument("--llm", type=str, default="gpt-oss:120b")



    args = parser.parse_args()
    if args.subject_folder:
        name = os.path.basename(os.path.normpath(args.subject_folder))
    else:
        name = args.root_folder
    logger = get_logger(name) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    main(args)