import os, logging
from pathlib import Path

import nibabel as nib
import numpy as np

from skimage.draw import line
from scipy.ndimage import binary_dilation
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


def interpolate_midline_rows(midline_path, t1_path, out_path, plane_axis=2, interp_axis=1):
    """
    Interpolates missing rows (all zeros) in a thin midline mask along the specified axis.
    For each 2D plane orthogonal to `plane_axis`, checks lines along `interp_axis`.
    If a line is empty but lies within the brainmask, it is interpolated between
    the nearest nonzero lines above and below.
    """
    mid = nib.load(midline_path)
    mid_data = mid.get_fdata().astype(float)
    brainmask = nib.load(t1_path).get_fdata() > 0

    if mid_data.shape != brainmask.shape:
        raise ValueError(f"Shape mismatch: midline {mid_data.shape}, brainmask {brainmask.shape}")

    repaired = mid_data.copy()
    shape = mid_data.shape

    # iterate over slices orthogonal to plane_axis
    for i in range(shape[plane_axis]):
        plane = np.take(repaired, i, axis=plane_axis)
        mask_plane = np.take(brainmask, i, axis=plane_axis)

        # move interpolation axis to front for convenience
        plane = np.moveaxis(plane, interp_axis if interp_axis < plane_axis else interp_axis - 1, 0)
        mask_plane = np.moveaxis(mask_plane, interp_axis if interp_axis < plane_axis else interp_axis - 1, 0)

        for j in range(plane.shape[0]):
            row = plane[j, :]
            if np.all(row == 0) and np.any(mask_plane[j, :]):
                above = next((jj for jj in range(j - 1, -1, -1) if np.any(plane[jj, :])), None)
                below = next(
                    (jj for jj in range(j + 1, plane.shape[0]) if np.any(plane[jj, :])),
                    None,
                )
                if above is not None and below is not None:
                    plane[j, :] = 0.5 * (plane[above, :] + plane[below, :])
                elif above is not None:
                    plane[j, :] = plane[above, :]
                elif below is not None:
                    plane[j, :] = plane[below, :]

        # move interpolation axis back and replace plane
        plane = np.moveaxis(plane, 0, interp_axis if interp_axis < plane_axis else interp_axis + 1)
        repaired = np.swapaxes(repaired, plane_axis, 2)
        repaired[:, :, i] = plane
        repaired = np.swapaxes(repaired, 2, plane_axis)

    repaired = (repaired > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(repaired, mid.affine, mid.header), out_path)
    logger.info(f"- Filled holes in midline, saved to {out_path}")


def ideal_midline_from_deformed(deformed_midline_path, ideal_midline_path, overwrite=False, label=1, fill_label=2):
    if not os.path.exists(ideal_midline_path) or overwrite:
        deformed_nii = nib.load(str(deformed_midline_path))
        mask = deformed_nii.get_fdata()

        connected = np.zeros_like(mask, dtype=np.uint8)
        X, Y, Z = mask.shape
        for z in range(Z):
            sl = mask[:, :, z]
            coords = np.argwhere(sl == label)
            if coords.shape[0] < 2:
                continue
            dists = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            (x1, y1), (x2, y2) = coords[i], coords[j]
            rr, cc = line(x1, y1, x2, y2)
            connected[rr, cc, z] = fill_label

        out_nii = nib.Nifti1Image(connected.astype(np.uint8), deformed_nii.affine, deformed_nii.header)
        nib.save(out_nii, str(ideal_midline_path))
        logger.info(f"- Saved ideal midline to: {ideal_midline_path}")
        return out_nii
    else:
        logger.info(f'Path {ideal_midline_path} exists, skipping ideal midline creation step...')


def select_regions(seg, regions):
    mask = np.isin(seg, regions)
    return seg * mask


def split_mask_by_midline(binary_midline, mask, expand_voxels=100):
    """
    Splits a mask into left and right hemisphere according to the provided binary midline.
    Works by expanding the midline plane in each direction by expand_voxels then calculating the overlap with the mask.
    Returns volumes on both left and right sides of the midline.
    """
    volumes = {}
    for direction in ["left", "right"]:
        extended_one_side = np.zeros_like(binary_midline, dtype=bool)
        for x, y, z in np.argwhere(binary_midline):
            if direction == "left":
                end = min(binary_midline.shape[0], x + expand_voxels)
                extended_one_side[x:end, y, z] = True
            else:
                start = max(0, x - expand_voxels)
                extended_one_side[start : x + 1, y, z] = True
        volumes[direction] = extended_one_side & mask
    return volumes


def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # convert np.int64, np.float32, etc.
    else:
        return obj


def update_midline(
    subject_midline_path,
    ideal_midline_path,
    tumor_mask_path,
    updated_midline_save_path,
    overwrite=False,
    crosses_threshold=200,
    verbose=False,
    ncr_label=1,
    ed_label=2,
    et_label=3,
):
    """
    Updates the midline to handle cases where the tumor extends across the anatomical midline.
    In such cases, the updated midline is defined as the boundary of the combined (non-enhancing core + enhancing tumor) region on the side opposite the tumor's dominant bulk.
    """
    subject_midline_im = nib.load(subject_midline_path)
    subject_midline = subject_midline_im.get_fdata()
    binary_midline = np.asarray(subject_midline) > 0

    binary_ideal_midline = nib.load(ideal_midline_path).get_fdata() > 0

    tumor_mask = nib.load(tumor_mask_path).get_fdata()
    binary_ncr = select_regions(tumor_mask, [0, ncr_label]) > 0
    binary_ed = select_regions(tumor_mask, [0, ed_label]) > 0
    binary_et = select_regions(tumor_mask, [0, et_label]) > 0
    binary_ncr_et = select_regions(tumor_mask, [0, ncr_label, et_label]) > 0

    overlaps = []
    volumes = []

    sides = ["left", "right"]

    data = defaultdict(dict)

    subregion_volumes = defaultdict(list)
    subregion_overlaps = defaultdict(list)
    smallest_volume = {}
    crosses_midline = {}

    subregion_volumes_ideal = defaultdict(list)
    subregion_overlaps_ideal = defaultdict(list)
    smallest_volume_ideal = {}
    subregions = {"ncr": binary_ncr, "ed": binary_ed, "et": binary_et, "ncr_et": binary_ncr_et}
    for region in subregions.keys():
        volumes = split_mask_by_midline(binary_midline=binary_midline, mask=subregions[region])
        volumes_ideal = split_mask_by_midline(binary_midline=binary_ideal_midline, mask=subregions[region])
        for direction in sides:  # left, right
            vol = volumes[direction]
            overlap = np.sum(vol)
            subregion_volumes[region].append(vol)
            subregion_overlaps[region].append(overlap)

            vol_ideal = volumes_ideal[direction]
            overlap_ideal = np.sum(vol_ideal)
            subregion_volumes_ideal[region].append(vol_ideal)
            subregion_overlaps_ideal[region].append(overlap_ideal)

        smallest_volume[region] = subregion_volumes[region][np.argmin(subregion_overlaps[region])]
        smallest_volume_ideal[region] = subregion_overlaps_ideal[region][np.argmin(subregion_overlaps[region])]
        if verbose:
            logger.info(
                f"Tumor size {region}, volumes {subregion_overlaps[region]}; Primary side: {sides[np.abs(1-np.argmin(subregion_overlaps[region]))]}"
            )
            logger.info(
                f"[ideal midline] Tumor size {region}, volumes {subregion_overlaps_ideal[region]}; Primary side: {sides[np.abs(1-np.argmin(subregion_overlaps_ideal[region]))]}"
            )
            logger.info(
                f"crosses midline {region}: {subregion_overlaps_ideal[region][np.argmin(subregion_overlaps_ideal[region])]>crosses_threshold}"
            )

        data["volumes_ideal_midline"][region] = dict(zip(sides, subregion_overlaps_ideal[region]))
        data["crosses_ideal_midline"][region] = subregion_overlaps_ideal[region][np.argmin(subregion_overlaps_ideal[region])] > crosses_threshold
        data["primary_side_ideal_midline"][region] = sides[np.argmax(subregion_overlaps_ideal[region])]

        data["volumes_patient_midline"][region] = dict(zip(sides, subregion_overlaps[region]))
        data["primary_side_patient_midline"][region] = sides[np.argmax(subregion_overlaps[region])]
        data["crosses_patient_midline"][region] = subregion_overlaps[region][np.argmin(subregion_overlaps[region])] > crosses_threshold

    dilated = binary_dilation(smallest_volume["ncr_et"], iterations=1)
    outer_shell = dilated & (~smallest_volume["ncr_et"])
    dilated_t = binary_dilation(binary_ncr_et, iterations=1)
    outer_shell_tumor = dilated_t & (~binary_ncr_et)
    updated_midline = ~(~(outer_shell_tumor & outer_shell)) | (binary_midline & ~binary_ncr_et)

    if not os.path.exists(updated_midline_save_path) or overwrite:
        nib.save(nib.Nifti1Image(updated_midline.astype(np.uint8), subject_midline_im.affine, subject_midline_im.header), updated_midline_save_path)
        logger.info(f'Updated midline to account for tumor... saved updated mask to {updated_midline_save_path}')
    else:
        logger.info(f'Path {updated_midline_save_path} exists, skipping midline updating save step...')

    return data


def merge_masks(masks):
    merged = np.zeros_like(masks[0])
    for m in masks:
        merged[m > 0] = m[m > 0]
    return merged


def connect_labels_plane(mask, label1=1, label2=2):
    """
    Connects label1 and label2 vertically along the y-axis for each (x, z) column.
    Fills voxels between them with line length and returns both:
      - connected mask
      - dictionary {z_index: [list of line lengths for that slice]}
    """
    connected = np.zeros_like(mask, dtype=float)
    line_lengths_by_slice = {}
    X, Y, Z = mask.shape

    for z in range(Z):
        slice_lengths = []
        for x in range(X):
            column = mask[:, x, z]
            y1 = np.where(column == label1)[0]
            y2 = np.where(column == label2)[0]
            if len(y1) and len(y2):
                y1c, y2c = int(round(y1.mean())), int(round(y2.mean()))
                y_min, y_max = sorted([y1c, y2c])
                line_len = y2c - y1c
                slice_lengths.append(line_len)
                connected[y_min : y_max + 1, x, z] = line_len
            # else:
            #     slice_lengths.append(0.0)
        if not slice_lengths:
            slice_lengths = [0.0]
        line_lengths_by_slice[z] = slice_lengths

    return connected, line_lengths_by_slice


def midline_distance_fill(ideal_midline_path, deformed_midline_path, midline_distances_path, overwrite=False, label_ideal=2, label_deformed=1, metadata=None, save_dir=None,):
    """Compute per-line midline distances and save both NIfTI and JSON summary."""
    ideal = nib.load(str(ideal_midline_path)).get_fdata()
    deformed = nib.load(str(deformed_midline_path)).get_fdata()

    merged = merge_masks([ideal, deformed])
    connected, line_lengths = connect_labels_plane(merged, label1=label_deformed, label2=label_ideal)

    ref_nii = nib.load(str(ideal_midline_path))
    out_nii = nib.Nifti1Image(connected.astype(np.float32), ref_nii.affine, ref_nii.header)

    if overwrite or not os.path.exists(midline_distances_path):
        nib.save(out_nii, str(midline_distances_path))
        logger.info(f"- Saved midline distance map to: {midline_distances_path}")
    else:
        logger.info(f'Path {midline_distances_path} exists, skipping midline distances save step...')

    # Pick per-slice signed max (by absolute magnitude)
    max_shift_per_slice = {int(z): float(v[np.argmax(np.abs(v))]) if len(v) else 0.0 for z, v in line_lengths.items()}

    # Compute summary statistics from signed per-slice shifts
    slice_signed_max_shifts = list(max_shift_per_slice.values())

    if slice_signed_max_shifts:
        mean_shift = float(np.mean(slice_signed_max_shifts))
        median_shift = float(np.median(slice_signed_max_shifts))
        max_shift = float(slice_signed_max_shifts[np.argmax(np.abs(slice_signed_max_shifts))])
        p95_shift = float(np.percentile(np.abs(slice_signed_max_shifts), 95))
    else:
        mean_shift = median_shift = max_shift = p95_shift = None



    summary = {
        # "midline_shift_file": str(midline_distances_path),
        "n_slices_with_shift": sum(bool(v and any(x > 0 for x in v)) for v in line_lengths.values()),
        "mean_shift_mm": mean_shift,
        "median_shift_mm": median_shift,
        "max_shift_mm": max_shift,
        "p95_shift_mm": p95_shift,
        # "max_shift_per_slice_mm": max_shift_per_slice,  # {slice_index: max shift}
    }
    if metadata:
        summary.update(metadata)

    summary = to_builtin(summary)

    if save_dir is not None:
        json_path = os.path.join(save_dir, "midline_statistics.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"- Saved midline distance summary to: {json_path}")
    return summary


def midline_shift_3d(tmp_dir, tumor, ncr_label=1, ed_label=2, et_label=4, overwrite=False):
    deformed_midline_path = os.path.join(tmp_dir, "patient_midline.nii.gz")
    ideal_midline_path = os.path.join(tmp_dir, "ideal_midline.nii.gz")
    midline_distances_path = os.path.join(tmp_dir, "midline_distances.nii.gz")
    logger.info(f"** [2/4] Starting midline shift processing...")

    ideal_midline_from_deformed(
        deformed_midline_path=deformed_midline_path,
        ideal_midline_path=ideal_midline_path,
        overwrite=overwrite,
    )

    if tumor is not None:
        metadata = update_midline(
            subject_midline_path=deformed_midline_path,
            ideal_midline_path=ideal_midline_path,
            tumor_mask_path=tumor,
            updated_midline_save_path=deformed_midline_path,
            ncr_label=ncr_label,
            ed_label=ed_label,
            et_label=et_label,
            overwrite=overwrite,
        )
    else:
        metadata = None

    summary = midline_distance_fill(
        ideal_midline_path=ideal_midline_path,
        deformed_midline_path=deformed_midline_path,
        midline_distances_path=midline_distances_path,
        metadata=metadata,
        overwrite=overwrite,
    )
    logger.info(f"* Finished processing midline shift! Saved results to {tmp_dir}")

    return summary


# python3 midline_shift3d.py --transform "/mmfs1/gscratch/kurtlab/brats2023/data/MNI152_IN_SUBJECT_SPACE/transforms/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000/BraTS-GLI-00134-000-t1n.nii.gz" --save_dir /mmfs1/gscratch/kurtlab/MSFT/metadata_analysis/midline/dev/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000 --tumor /mmfs1/gscratch/kurtlab/brats2023/data/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000/BraTS-GLI-00134-000-seg.nii.gz


# python3 midline_shift3d.py --t1 /mmfs1/gscratch/kurtlab/brats2023/data/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00009-001/BraTS-GLI-00009-001-t1n.nii.gz --save_dir /mmfs1/gscratch/kurtlab/MSFT/metadata_analysis/midline/dev/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00009-001
# python3 midline_shift3d.py --transform "/mmfs1/gscratch/kurtlab/brats2023/data/MNI152_IN_SUBJECT_SPACE/transforms/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000/BraTS-GLI-00134-000-t1n.nii.gz" --save_dir /mmfs1/gscratch/kurtlab/MSFT/metadata_analysis/midline/dev/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000_v0
# python3 midline_shift3d.py --transform "/mmfs1/gscratch/kurtlab/brats2023/data/MNI152_IN_SUBJECT_SPACE/transforms/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000/BraTS-GLI-00134-000-t1n.nii.gz" --save_dir /mmfs1/gscratch/kurtlab/MSFT/metadata_analysis/midline/dev/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000 --tumor /mmfs1/gscratch/kurtlab/brats2023/data/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000/BraTS-GLI-00134-000-seg.nii.gz

# python3 midline_shift3d.py --save_dir /mmfs1/gscratch/kurtlab/MSFT/metadata_analysis/midline/dev/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000 --tumor /mmfs1/gscratch/kurtlab/brats2023/data/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000/BraTS-GLI-00134-000-seg.nii.gz --t1 '/mmfs1/gscratch/kurtlab/brats2023/data/brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00134-000/BraTS-GLI-00134-000-t1n.nii.gz'
