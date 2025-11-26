import os, subprocess, argparse, logging
from scipy.ndimage import binary_dilation
import nibabel as nib
import numpy as np


logger = logging.getLogger(__name__)


WRAPPER = os.environ.get("SYNTHSEG_SIF")
if WRAPPER is None:
    logger.error("Environment variable SYNTHMORPH_SIF pointing to SynthMorph .sif is not set!")
    raise RuntimeError("Environment variable SYNTHSEG_SIF pointing to SynthSeg .sif is not set!")


# def synthseg(input_path, output_path, parc=False, robust=False, fast=False, cpu=False, wrapper=WRAPPER, overwrite=False):
#     if os.path.exists(output_path) and not overwrite:
#         logger.info(f"Path {output_path} exists, skipping segmentation..")
#         return
#     """Run SynthSeg segmentation for one subject."""
#     logger.info(f"========================================")
#     logger.info(f"    SynthSeg Segmentation")
#     logger.info(f"----------------------------------------")
#     logger.info(f"  Input    : {input_path}")
#     logger.info(f"  Output    : {output_path}")
#     logger.info(f"========================================")

#     cmd = [wrapper, "--i", input_path, "--o", output_path]
#     if parc:
#         cmd.append("--parc")
#     if robust:
#         cmd.append("--robust")
#     if fast:
#         cmd.append("--fast")
#     if cpu:
#         cmd.append("--cpu")
#     subprocess.run(cmd, check=True)

def synthseg(input_path, output_path, parc=False, robust=False, fast=False, cpu=False, sif=WRAPPER):
    if os.path.exists(output_path):
        return

    cmd = [
        "apptainer", "exec", "--nv",
        sif,
        "python", "/opt/SynthSeg/scripts/commands/SynthSeg_predict.py",
        "--i", input_path,
        "--o", output_path
    ]


    if parc:
        cmd.append("--parc")
    if robust:
        cmd.append("--robust")
    if fast:
        cmd.append("--fast")
    if cpu:
        cmd.append("--cpu")

    subprocess.run(cmd, check=True)


TUMOR_LABEL_MAPS = {
    "brats-men": {1: 64, 2: 65, 3: 66},  # Meningioma NCR, ED, ET
    "brats-ped": {1: 67, 2: 68, 3: 69},  # Pediatric Glioma NCR, ED, ET
    "brats-gli": {1: 61, 2: 62, 3: 63},  # Adult Glioma NCR, ED, ET
}


def update_tumor_labels(image_array: np.ndarray, label_map: dict):
    new_array = image_array.copy()
    for old_label, new_label in label_map.items():
        new_array[image_array == old_label] = new_label
    return new_array.astype(np.uint8)


def merge_tumor_midline_and_anat_masks(tumor_path, synthseg_path, midline_path, save_path, overwrite=False, tumor_type="glioma", ncr_label=1, ed_label=2, et_label=3):
    
    if os.path.exists(save_path) and not overwrite:
        logger.info(f'Path {save_path} exists, skipping merging of anatomical, midline, and tumor segmentations..')
        return
    TUMOR_LABEL_MAPS = {
        "meningioma": {ncr_label: 64, ed_label: 65, et_label: 66},  # Meningioma NCR, ED, ET
        "pediatric-glioma": {ncr_label: 67, ed_label: 68, et_label: 69},  # Pediatric Glioma NCR, ED, ET
        "glioma": {ncr_label: 61, ed_label: 62, et_label: 63},  # Adult Glioma NCR, ED, ET
    }

    tumor_im = nib.load(tumor_path)
    tumor = tumor_im.get_fdata()
    midline = nib.load(midline_path).get_fdata()
    synthseg = nib.load(synthseg_path).get_fdata()
    brainmask = (synthseg > 0).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for tt in TUMOR_LABEL_MAPS.keys():
        if tt.lower() in str(tumor_path).lower():
            tumor_type = tt

    midline = (midline > 0) * 70
    midline = binary_dilation(midline > 0, iterations=1).astype(np.uint8) * 70
    midline[synthseg == 0] = 0

    merged = synthseg.copy()
    tumor = update_tumor_labels(image_array=tumor, label_map=TUMOR_LABEL_MAPS[tumor_type])
    merged[tumor != 0] = tumor[tumor != 0]
    merged[midline != 0] = midline[midline != 0]
    merged[brainmask == 0] = 0   

    merged = np.rint(merged).astype(np.uint8)
    nib.save(nib.Nifti1Image(merged, affine=tumor_im.affine, header=tumor_im.header), save_path)
    logger.info('Merged anatomical, midline, and tumor segmentations successfully!')


"""

Index for merged Anatomical, Midline, and Tumor masks. 

The first 60 slots come from anatomical regions from SynthSeg 2.0
    0:  background
    1:  empty (reserved for addtl masks like brainmask)

    2:  left cerebral white matter
    3:  left cerebral cortex
    4:  left lateral ventricle
    5:  left inferior lateral ventricle
    7:  left cerebellum white matter
    8:  left cerebellum cortex
    10: left thalamus
    11: left caudate
    12: left putamen
    13: left pallidum
    14: 3rd ventricle
    15: 4th ventricle
    16: brain-stem
    17: left hippocampus
    18: left amygdala
    24: CSF
    26: left accumbens area
    28: left ventral DC
    41: right cerebral white matter
    42: right cerebral cortex
    43: right lateral ventricle
    44: right inferior lateral ventricle
    46: right cerebellum white matter
    47: right cerebellum cortex
    49: right thalamus
    50: right caudate
    51: right putamen
    52: right pallidum
    53: right hippocampus
    54: right amygdala
    58: right accumbens area
    60: right ventral DC

The last 10 slots come from BraTS labels and the patient midline
    61: Glioma NCR
    62: Glioma ED
    63: Glioma ET
    64: Meningioma NCR
    65: Meningioma ED
    66: Meningioma ET
    67: Pediatric Glioma NCR
    68: Pediatric Glioma ED
    69: Pediatric Glioma ET

    70: Patient Midline

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--parc", required=False, action="store_true")
    parser.add_argument("--robust", required=False, action="store_true")
    parser.add_argument("--fast", required=False, action="store_true")
    parser.add_argument("--cpu", required=False, action="store_true")

    parser.add_argument("--wrapper", default=WRAPPER, required=False)
    args = parser.parse_args()
    synthseg(input_path=args.input, output_path=args.output, parc=args.parc, robust=args.robust, fast=args.fast, cpu=args.cpu, wrapper=WRAPPER)
