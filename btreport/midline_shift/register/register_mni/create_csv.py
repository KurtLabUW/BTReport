import os
import glob
import csv
import math


FIXED = "/gscratch/kurtlab/MSFT/metadata_analysis/register_mni/MNI152_T1_1mm_Brain.nii.gz"
SUBJECTS_DIR = "/gscratch/kurtlab"
pathologies = {
    "glioma": ["brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", "brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"],
    "meningioma": ["brats-men/ASNR-MICCAI-BraTS2023-Meningioma-Challenge-TrainingData", "brats-men/ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData"],
    "pediatric-glioma": ["brats-ped/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"],
}


"""
paths_ref:
export SUBJECTS_DIR="/gscratch/kurtlab"
"$WRAPPER" register -g   -o "MSFT/metadata_analysis/register_mni/moved/BraTS-GLI-00002-000-t1n.nii.gz"   -t "MSFT/metadata_analysis/register_mni/transforms/BraTS-GLI-00002-000-t1n.nii.gz"   "brats2023/data/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00002-000/BraTS-GLI-00002-000-t1n.nii.gz"   "MSFT/metadata_analysis/register_mni/MNI152_T1_1mm_Brain.nii.gz"
"""


def create_paths_ref(
    pathology_list=["glioma", "meningioma", "pediatric-glioma"],
    ending="-t1n.nii.gz",
    root_moving="/gscratch/kurtlab/brats2023/data",
    root_moved="/gscratch/kurtlab/brats2023/data/MNI152",
    csv_save_path="./paths_ref.csv",
    remove_subject_dir=True,
):
    # gather moving, fixed, moved, transform file paths
    ALL_MOVING = []
    ALL_TRANSFORM = []
    ALL_MOVED = []
    for pathology in pathology_list:
        for path in pathologies[pathology]:
            subject_names = os.listdir(os.path.join(root_moving, path))
            moving_paths = [os.path.join(root_moving, path, f, f"{f}{ending}") for f in subject_names]
            moved_paths = []
            transform_paths = []

            for f in subject_names:
                moved_dir = os.path.join(root_moved, "moved", path, f)
                transform_dir = os.path.join(root_moved, "transforms", path, f)
                os.makedirs(moved_dir, exist_ok=True)
                os.makedirs(transform_dir, exist_ok=True)
                moved_paths.append(os.path.join(moved_dir, f"{f}{ending}"))
                transform_paths.append(os.path.join(transform_dir, f"{f}{ending}"))

            ALL_MOVING += moving_paths
            ALL_MOVED += moved_paths
            ALL_TRANSFORM += transform_paths

    # make all fixed files the same
    ALL_FIXED = [FIXED] * len(ALL_MOVING)

    if remove_subject_dir:
        def strip_prefix(p):
            p = os.path.abspath(p)
            if p.startswith(SUBJECTS_DIR):
                return p.replace(SUBJECTS_DIR + "/", "")
            return p
        ALL_MOVING = [strip_prefix(p) for p in ALL_MOVING]
        ALL_MOVED = [strip_prefix(p) for p in ALL_MOVED]
        ALL_TRANSFORM = [strip_prefix(p) for p in ALL_TRANSFORM]
        ALL_FIXED = [strip_prefix(p) for p in ALL_FIXED]


    with open(csv_save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["moving", "fixed", "moved", "transform"])  # header row
        for m, fxd, md, tf in zip(ALL_MOVING, ALL_FIXED, ALL_MOVED, ALL_TRANSFORM):
            writer.writerow([m, fxd, md, tf])

    print(f"Saved {len(ALL_MOVING)} pairs to {csv_save_path}")


def create_paths_apply_transform(
    pathology_list=["glioma", "meningioma", "pediatric-glioma"],
    ref_ending="-t1n.nii.gz",
    target_endings=["-t2f.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-seg.nii.gz"],
    root_moving="brats2023/data",
    root_moved="brats2023/data/MNI152",
    paths_ref_csv="./paths_ref.csv",
    csv_save_path='./paths_apply_transform.csv'
):
    ALL_TRANSFORM=[]
    ALL_MOVING=[]
    ALL_MOVED=[]

    # Load all CSV rows (skip header)
    with open(paths_ref_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows):
        for ending in target_endings:
            moving_ref, _, _, transform = (row[k].strip() for k in row)
            moving = moving_ref.replace(ref_ending, ending)
            moved = moving.replace(root_moving, root_moved)
            
            ALL_TRANSFORM.append(transform)
            ALL_MOVING.append(moving)
            ALL_MOVED.append(moved)


    with open(csv_save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transform", "moving", "moved"])  # header row
        for tf, m, md in zip(ALL_TRANSFORM, ALL_MOVING, ALL_MOVED):
            writer.writerow([tf, m, md])

    print(f"Saved {len(ALL_MOVING)} pairs to {csv_save_path}")


def create_paths_atlas_to_subject(
    pathology_list=["glioma", "meningioma", "pediatric-glioma"],
    ending="-t1n.nii.gz",
    root_moving="/gscratch/kurtlab/brats2023/data",
    root_moved="/gscratch/kurtlab/brats2023/data/MNI152_IN_SUBJECT_SPACE",
    csv_save_path="./paths_atlas_to_subject.csv",
    remove_subject_dir=True,
):
    # gather moving, fixed, moved, transform file paths
    ALL_MOVING = []     
    ALL_TRANSFORM = []
    ALL_MOVED = []
    ALL_FIXED=[]

    for pathology in pathology_list:
        for path in pathologies[pathology]:
            subject_names = os.listdir(os.path.join(root_moving, path))

            # subject images are now FIXED, atlas is MOVING
            fixed_paths = [os.path.join(root_moving, path, f, f"{f}{ending}") for f in subject_names]
            moved_paths = []
            transform_paths = []

            for f in subject_names:
                moved_dir = os.path.join(root_moved, "moved", path, f)
                transform_dir = os.path.join(root_moved, "transforms", path, f)
                os.makedirs(moved_dir, exist_ok=True)
                os.makedirs(transform_dir, exist_ok=True)
                moved_paths.append(os.path.join(moved_dir, f"{f}{ending}"))
                transform_paths.append(os.path.join(transform_dir, f"{f}{ending}"))

            ALL_MOVED += moved_paths
            ALL_TRANSFORM += transform_paths

            # same atlas for all subjects (atlas moves to subject space)
            ALL_MOVING += [FIXED] * len(subject_names)
            ALL_FIXED += fixed_paths

    if remove_subject_dir:
        def strip_prefix(p):
            p = os.path.abspath(p)
            if p.startswith(SUBJECTS_DIR):
                return p.replace(SUBJECTS_DIR + "/", "")
            return p
        ALL_MOVING = [strip_prefix(p) for p in ALL_MOVING]
        ALL_FIXED = [strip_prefix(p) for p in ALL_FIXED]
        ALL_MOVED = [strip_prefix(p) for p in ALL_MOVED]
        ALL_TRANSFORM = [strip_prefix(p) for p in ALL_TRANSFORM]

    with open(csv_save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["moving", "fixed", "moved", "transform"])
        for m, fxd, md, tf in zip(ALL_MOVING, ALL_FIXED, ALL_MOVED, ALL_TRANSFORM):
            writer.writerow([m, fxd, md, tf])

    print(f"Saved {len(ALL_MOVING)} atlas→subject pairs to {csv_save_path}")


def check_if_ref_complete(
    pathology_list=["glioma", "meningioma", "pediatric-glioma"],
    ending="-t1n.nii.gz",
    root_moving="/gscratch/kurtlab/brats2023/data",
    root_moved="/gscratch/kurtlab/brats2023/data/MNI152",
    csv_save_path="./paths_ref.csv",
    remove_subject_dir=True,
):
    # gather moving, fixed, moved, transform file paths
    ALL_MOVING = []
    ALL_TRANSFORM = []
    ALL_MOVED = []
    for pathology in pathology_list:
        for path in pathologies[pathology]:
            subject_names = os.listdir(os.path.join(root_moving, path))
            moving_paths = [os.path.join(root_moving, path, f, f"{f}{ending}") for f in subject_names]
            moved_paths = []
            transform_paths = []

            for f in subject_names:
                moved_dir = os.path.join(root_moved, "moved", path, f)
                transform_dir = os.path.join(root_moved, "transforms", path, f)
                os.makedirs(moved_dir, exist_ok=True)
                os.makedirs(transform_dir, exist_ok=True)
                moved_paths.append(os.path.join(moved_dir, f"{f}{ending}"))
                transform_paths.append(os.path.join(transform_dir, f"{f}{ending}"))

            ALL_MOVING += moving_paths
            ALL_MOVED += moved_paths
            ALL_TRANSFORM += transform_paths

        ALL_MOVED= sorted(ALL_MOVED)
        success = [os.path.exists(m) for m in ALL_MOVED]
        raise ValueError('expected: ',len(sorted(ALL_MOVED)), 'actual: ',sum(success))


def check_if_atlas_complete(
    pathology_list=["glioma", "meningioma", "pediatric-glioma"],
    ending="-t1n.nii.gz",
    root_moving="/gscratch/kurtlab/brats2023/data",
    root_moved="/gscratch/kurtlab/brats2023/data/MNI152_IN_SUBJECT_SPACE",
    # csv_save_path="./paths_atlas_to_subject.csv",
    remove_subject_dir=True,
):
    # gather moving, fixed, moved, transform file paths
    ALL_MOVING = []     
    ALL_TRANSFORM = []
    ALL_MOVED = []
    ALL_FIXED=[]

    for pathology in pathology_list:
        for path in pathologies[pathology]:
            subject_names = os.listdir(os.path.join(root_moving, path))

            # subject images are now FIXED, atlas is MOVING
            fixed_paths = [os.path.join(root_moving, path, f, f"{f}{ending}") for f in subject_names]
            moved_paths = []
            transform_paths = []

            for f in subject_names:
                moved_dir = os.path.join(root_moved, "moved", path, f)
                transform_dir = os.path.join(root_moved, "transforms", path, f)
                os.makedirs(moved_dir, exist_ok=True)
                os.makedirs(transform_dir, exist_ok=True)
                moved_paths.append(os.path.join(moved_dir, f"{f}{ending}"))
                transform_paths.append(os.path.join(transform_dir, f"{f}{ending}"))

            ALL_MOVED += moved_paths
            ALL_TRANSFORM += transform_paths

            # same atlas for all subjects (atlas moves to subject space)
            ALL_MOVING += [FIXED] * len(subject_names)
            ALL_FIXED += fixed_paths
    
        ALL_MOVED= sorted(ALL_MOVED)
        success = [os.path.exists(m) for m in ALL_MOVED]
        raise ValueError('expected: ',len(sorted(ALL_MOVED)), 'actual: ',sum(success))


# print("\n".join([moving, fixed, moved, transform]))
# "$WRAPPER" apply -m nearest "MSFT/metadata_analysis/register_mni/transforms/BraTS-GLI-00002-000-t1n.nii.gz"  "brats2023/data/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00002-000/BraTS-GLI-00002-000-seg.nii.gz" "MSFT/metadata_analysis/register_mni/moved/BraTS-GLI-00002-000-seg.nii.gz"
# "$WRAPPER" apply -m nearest $TRANSFORM  $MOVING $MOVED

if __name__ == "__main__":
    # create_paths_ref()
    # create_paths_apply_transform()
    # create_paths_atlas_to_subject()
    # check_if_ref_complete()
    check_if_atlas_complete()


