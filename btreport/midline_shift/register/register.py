import csv, sys, os, math, subprocess, datetime, argparse, logging

logger = logging.getLogger(__name__)

MNI152 = "/gscratch/kurtlab/MSFT/metadata_analysis/register_mni/MNI152_T1_1mm_Brain.nii.gz"

SUBJECTS_DIR = os.environ.get("SUBJECTS_DIR")
if SUBJECTS_DIR is None:
    logger.info("Environment variable SUBJECTS_DIR working dir is not set, defaulting to curent working dir")
    SUBJECTS_DIR = os.getcwd()
    os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR


WRAPPER = os.environ.get("WRAPPER")
if WRAPPER is None:
    raise RuntimeError("Environment variable WRAPPER pointing to SynthMorph .sif is not set!")


def run_registration(moving, fixed, moved, transform, wrapper=WRAPPER, subjects_dir=SUBJECTS_DIR):
    """Run SynthMorph registration for one subject."""
    print(f"\n========================================")
    print(f"    SynthMorph Registration")
    print(f"----------------------------------------")
    print(f"  Moving    : {os.path.join(subjects_dir,moving) if not os.path.isabs(moving) else moving}")
    print(f"  Fixed     : {os.path.join(subjects_dir,fixed) if not os.path.isabs(fixed) else fixed}")
    print(f"  Output    : {os.path.join(subjects_dir,moved) if not os.path.isabs(moved) else moved}")
    print(f"  Transform : {os.path.join(subjects_dir,transform) if not os.path.isabs(transform) else transform}")
    print(f"========================================")

    cmd = [wrapper, "register", "-g",
           "-o", moved, "-t", transform,
           moving, fixed]
    subprocess.run(cmd, check=True)


def run_apply(transform, moving, moved, wrapper, subjects_dir, is_seg=False):
    """Run SynthMorph apply for one image or segmentation."""
    print(f"\n========================================")
    print(f"    SynthMorph Apply Registration")
    print(f"----------------------------------------")
    print(f"  Moving    : {os.path.join(subjects_dir,moving)}")
    print(f"  Output    : {os.path.join(subjects_dir,moved)}")
    print(f"  Transform : {os.path.join(subjects_dir,transform)}")
    print(f"========================================")
    cmd = [wrapper, "apply"]
    if is_seg:
        cmd += ["-m", "nearest"]
    cmd += [transform, moving, moved]
    subprocess.run(cmd, check=True)



def register_to_mni(moving, moved, transform, subjects_dir=SUBJECTS_DIR, fixed=MNI152):
    run_registration(moving=moving, fixed=fixed, moved=moved, transform=transform, 
                     wrapper=WRAPPER, subjects_dir=subjects_dir)

def register_mni_to_subject(fixed, moved, transform, subjects_dir=SUBJECTS_DIR, moving=MNI152):
    run_registration(moving=moving, fixed=fixed, moved=moved, transform=transform, 
                     wrapper=WRAPPER, subjects_dir=subjects_dir)


def apply_transform(moving, moved, transform, is_seg=False, wrapper=WRAPPER, subjects_dir=SUBJECTS_DIR):
    run_apply(transform=transform, moving=moving, moved=moved, wrapper=wrapper, subjects_dir=subjects_dir, is_seg=is_seg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--moving", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moved", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--wrapper", default=WRAPPER, required=False)
    parser.add_argument("--subjects_dir", default=SUBJECTS_DIR, required=False)
    args = parser.parse_args()
    run_registration(moving=args.moving, fixed=args.fixed, moved=args.moved, transform=args.transform, 
                     wrapper=args.wrapper, subjects_dir=args.subjects_dir)