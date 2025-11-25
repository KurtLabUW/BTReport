# This is the codebase for VASARI-auto
# -------------------------------------------------------------------------
# vasari-auto.py | a pipeline for automated VASARI characterisation of glioma.
# Copyright 2024 James Ruffle, High-Dimensional Neurology,
# UCL Queen Square Institute of Neurology.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# Not intended for clinical use.
#
# Original repository:
#     https://github.com/james-ruffle/vasari-auto
# Correspondence:
#     Dr James K Ruffle — j.ruffle@ucl.ac.uk
# -------------------------------------------------------------------------

# --- Modifications by Juampablo Heras Rivera for BTReport----------------------------------------
# This file has been modified from the original VASARI-auto implementation.
# Changes include:
#   - Improved midline crossing logic in the original subject space
#   - Multifocal or Multicentric logic: used connected components to get the number of tumor bodies, then only kept mutliple lesions if the non-largest lesions are >1cm.  Lesion is multifocal when more than one of these lesions exists
#   - Used the images image in their original space instead of the version registered MNI152 space to better estimate quantities like proportion necrotic, etc.. This was noted as an option in the original paper, but they used the MNI152 version.
#   - Added eloquent regions obtained from Brodmann Area Maps (https://surfer.nmr.mgh.harvard.edu/fswiki/BrodmannAreaMaps)
#   - Added lesion sizes APxTVxCC
#   - Adjusted to only include quantities that can be derived from BraTS masks. Removed proportion
# -------------------------------------------------------------------------


# Import packages
import numpy as np
import os, logging
import pandas as pd
import nibabel as nib
from scipy.ndimage import label
from sklearn.metrics import *
import time
from skimage.morphology import skeletonize

pd.set_option("display.max_rows", 500)
import math

logger = logging.getLogger(__name__)


def process_midline_stats(metadata):
    bool_to_index = {True: 3, False: 2}
    oedema_cross_midline_f = bool_to_index[metadata["crosses_ideal_midline"]["ed"]]
    nCET_cross_midline_f = bool_to_index[metadata["crosses_ideal_midline"]["ed"] or metadata["crosses_ideal_midline"]["ncr"]]
    CET_cross_midline_f = bool_to_index[metadata["crosses_ideal_midline"]["et"]]
    return oedema_cross_midline_f, nCET_cross_midline_f, CET_cross_midline_f


def get_vasari_features(
    file,
    file_ss=None,
    atlases="/gscratch/kurtlab/MSFT/metadata_analysis/vasari/atlas_masks/",
    merged=None,
    metadata=None,
    verbose=False,
    enhancing_label=3,
    nonenhancing_label=1,
    oedema_label=2,
    z_dim=-1,
    cf=1,
    t_ependymal=5000,
    t_wm=100,
    resolution=1,
    midline_thresh=5,
    enh_quality_thresh=15,
    cyst_thresh=50,
    cortical_thresh=1000,
    eloquent_thresh=100,
    focus_thresh=30000,
    num_components_bin_thresh=10,
    num_components_cet_thresh=15,
):
    """
    #Required argument
    file - NIFTI segmentation file with BraTS lesion labels in subject's native space
    atlases - atlas path for location derivation

    #Optional parameters
    file_ss - NIFTI segmentation file with BraTS lesion labels in subject's native space
    verbose - whether to enable verbose logging, default=False
    enhancing_label - the integer value of enhancing tumour within file, default=3
    nonenhancing_label - the integer value of nonenhancing tumour within file, default=1
    oeedema_label - the integer value of nonenhancing tumour within file, default=2
    z_dim - the dimension of the Z axis within file, default=-1, which assumes MNI template registration
    cf - correction factor for ambiguity in voxel quantification, default=1
    t_ependymal - threshold for lesion involvement within the ependyma, this can be customised depending on the voxel resolution you are operating in, default=5000
    t_wm - threshold for lesion involvement within the wm, this can be customised depending on the voxel resolution you are operating in, default=100
    resolution - volumetric voxel resolution, this is important for derivation of F11 - thickness of enhancing margin, default=1 (1mm x 1mm x 1mm resolution)
    midline_thresh - threshold for number of diseased voxels that can cross the midline to be quantified as a lesion definitively crossing the midline, default=5
    enh_quality_thresh - threshold for determining the quality of lesion enhancement by volume approximation. Please note ideally this feature would utilise source imaging but in its prototype format uses anonymised segmentation data only, default=15
    cyst_thresh - threshold for determining the presence of cysts based on a heuristic of nCET detection, default=50
    cortical_thresh - threshold for determining cortex involvement, default=1000
    focus_thresh - threshold for determining a principle side of involvement, this will vary depending on resolution, default=30000
    num_components_bin_thresh - threshold for quantifying a multifocal lesion, default = 10
    num_components_cet_thresh - threshold for satellite lesions, default=15
    """

    start_time = time.time()

    if verbose:
        logger.debug(
            "Please note that this software is in beta and utilises only irrevocably anonymised lesion masks.\nVASARI features that require source data shall not be derived and return NaN in this software version"
        )
        logger.debug("")
        logger.debug("Working on: " + str(file))
        logger.debug("")

    ##derive anatomy masks - this is for automated location (F1)
    brainstem = atlases + "brainstem.nii.gz"
    brainstem = nib.load(brainstem)
    brainstem_array = np.asanyarray(brainstem.dataobj)
    brainstem_vol = np.sum(brainstem_array)

    midline = atlases + "midline.nii.gz"
    midline = nib.load(midline)
    midline_array = np.asanyarray(midline.dataobj)
    midline_vol = np.sum(midline_array)

    frontal_lobe = atlases + "frontal_lobe.nii.gz"
    frontal_lobe = nib.load(frontal_lobe)
    frontal_lobe_array = np.asanyarray(frontal_lobe.dataobj)
    frontal_lobe_vol = np.sum(frontal_lobe_array)

    insula = atlases + "insula.nii.gz"
    insula = nib.load(insula)
    insula_array = np.asanyarray(insula.dataobj)
    insula_vol = np.sum(insula_array)

    occipital = atlases + "occipital.nii.gz"
    occipital = nib.load(occipital)
    occipital_array = np.asanyarray(occipital.dataobj)
    occipital_vol = np.sum(occipital_array)

    parietal = atlases + "parietal.nii.gz"
    parietal = nib.load(parietal)
    parietal_array = np.asanyarray(parietal.dataobj)
    parietal_vol = np.sum(parietal_array)

    temporal = atlases + "temporal.nii.gz"
    temporal = nib.load(temporal)
    temporal_array = np.asanyarray(temporal.dataobj)
    temporal_vol = np.sum(temporal_array)

    thalamus = atlases + "thalamus.nii.gz"
    thalamus = nib.load(thalamus)
    thalamus_array = np.asanyarray(thalamus.dataobj)
    thalamus_vol = np.sum(thalamus_array)

    corpus_callosum = atlases + "corpus_callosum.nii.gz"
    corpus_callosum = nib.load(corpus_callosum)
    corpus_callosum_array = np.asanyarray(corpus_callosum.dataobj)
    corpus_callosum_vol = np.sum(corpus_callosum_array)

    ventricles = atlases + "ventricles.nii.gz"
    ventricles = nib.load(ventricles)
    ventricles_array = np.asanyarray(ventricles.dataobj)
    ventricles_vol = np.sum(ventricles_array)

    internal_capsule = atlases + "internal_capsule.nii.gz"
    internal_capsule = nib.load(internal_capsule)
    internal_capsule_array = np.asanyarray(internal_capsule.dataobj)
    internal_capsule_vol = np.sum(internal_capsule_array)

    cortex = atlases + "cortex.nii.gz"
    cortex = nib.load(cortex)
    cortex_array = np.asanyarray(cortex.dataobj)
    cortex_vol = np.sum(cortex_array)

    eloquent = atlases + "eloquent_grouped.nii.gz"
    eloquent = nib.load(eloquent)
    eloquent_array = np.asanyarray(eloquent.dataobj)
    eloquent_array_mask = (eloquent_array > 0).astype(np.uint8)
    eloquent_vol = np.sum(eloquent_array_mask)

    segmentation = nib.load(file)
    segmentation_array = np.asanyarray(segmentation.dataobj)

    file_ss = file_ss if file_ss is not None else file
    segmentation_ss = nib.load(file_ss)
    segmentation_array_ss = np.asanyarray(segmentation_ss.dataobj)

    # if verbose:
    #     logger.debug('Running voxel quantification per tissue class')
    # total_lesion_burden = np.count_nonzero(segmentation_array)
    # enhancing_voxels = np.count_nonzero(segmentation_array == enhancing_label)
    # nonenhancing_voxels = np.count_nonzero(segmentation_array == nonenhancing_label)
    # oedema_voxels = np.count_nonzero(segmentation_array == oedema_label)

    if verbose:
        logger.debug("Running voxel quantification per tissue class")
    total_lesion_burden = np.count_nonzero(segmentation_array_ss)
    enhancing_voxels = np.count_nonzero(segmentation_array_ss == enhancing_label)
    nonenhancing_voxels = np.count_nonzero(segmentation_array_ss == nonenhancing_label)
    oedema_voxels = np.count_nonzero(segmentation_array_ss == oedema_label)

    # if verbose:
    # logger.debug('Deriving number of components')
    # labeled_array, num_components = label(segmentation_array)

    if verbose:
        logger.debug("Deriving number of components")
    labeled_array, num_components = label(segmentation_array_ss)

    if verbose:
        logger.debug("Determining laterality")
        # logger.debug('Note - if experiencing unexpected axis flipping for lesion laterality, check lesion registration space. This code assumes MNI template registration')
    temp = segmentation_array.nonzero()[0]
    right_hemisphere = len(temp[temp < int(segmentation_array.shape[z_dim] / 2)])
    left_hemisphere = len(temp[temp > int(segmentation_array.shape[z_dim] / 2)])
    if right_hemisphere > left_hemisphere:
        side = "Right"
    if right_hemisphere < left_hemisphere:
        side = "Left"
    if right_hemisphere > focus_thresh and left_hemisphere > focus_thresh:
        side = "Bilateral"
    # if verbose:
    # logger.debug(right_hemisphere)
    # logger.debug(left_hemisphere)

    if verbose:
        logger.debug("Determining proportions")
    segmentation_array[segmentation_array == oedema_label] = 0
    segmentation_array[segmentation_array == enhancing_label] = 1
    segmentation_array[segmentation_array == nonenhancing_label] = 1

    if segmentation_array.sum() == 0:
        if verbose:
            logger.debug("No lesion detected, falling back to oedema label for closer inspection")
        segmentation_array = np.asanyarray(segmentation.dataobj)
        segmentation_array[segmentation_array != oedema_label] = 0

    prop_in_brainstem = len((segmentation_array * brainstem_array).nonzero()[0]) / (segmentation_array.sum() + cf)
    prop_in_frontal_lobe = len((segmentation_array * frontal_lobe_array).nonzero()[0]) / (segmentation_array.sum() + cf)
    prop_in_insula = len((segmentation_array * insula_array).nonzero()[0]) / (segmentation_array.sum() + cf)
    prop_in_occipital = len((segmentation_array * occipital_array).nonzero()[0]) / (segmentation_array.sum() + cf)
    prop_in_parietal = len((segmentation_array * parietal_array).nonzero()[0]) / (segmentation_array.sum() + cf)
    prop_in_temporal = len((segmentation_array * temporal_array).nonzero()[0]) / (segmentation_array.sum() + cf)
    prop_in_thalamus = len((segmentation_array * thalamus_array).nonzero()[0]) / (segmentation_array.sum() + cf)
    prop_in_cc = len((segmentation_array * corpus_callosum_array).nonzero()[0]) / (segmentation_array.sum() + cf)

    d = {
        "ROI": ["Brainstem", "Frontal Lobe", "Insula", "Occipital Lobe", "Parietal Lobe", "Temporal Lobe", "Thalamus", "Corpus callosum"],
        "prop": [
            prop_in_brainstem,
            prop_in_frontal_lobe,
            prop_in_insula,
            prop_in_occipital,
            prop_in_parietal,
            prop_in_temporal,
            prop_in_thalamus,
            prop_in_cc,
        ],
    }

    vols = pd.DataFrame(data=d)
    vols = vols.sort_values(by="prop", ascending=False).reset_index(drop=True)

    region_prop_list = [(row.ROI, row.prop) for _, row in vols.iterrows() if row.prop >= 0.1]  # >10%

    if verbose:
        logger.debug(vols)
        logger.debug("Regions with >=10% tumor involvement:")
        logger.debug(region_prop_list)

    proportion_enhancing = (enhancing_voxels / (total_lesion_burden + 0.1)) * 100
    proportion_nonenhancing = (nonenhancing_voxels / (total_lesion_burden + 0.1)) * 100
    proportion_oedema = ((oedema_voxels / (total_lesion_burden + 0.1))) * 100

    enhancement_quality = 1
    if proportion_enhancing > 0:  # heuristic of if model segments more than 10% voxels are enhancing
        if proportion_enhancing > enh_quality_thresh:
            enhancement_quality = 3
        else:
            enhancement_quality = 2

    if verbose:
        logger.debug("Determining ependymal involvement")
    if len((segmentation_array * ventricles_array).nonzero()[0]) >= t_ependymal:
        ependymal = 2

    if len((segmentation_array * ventricles_array).nonzero()[0]) < t_ependymal:
        ependymal = 1

    if verbose:
        logger.debug("Determining white matter involvemenet")
    deep_wm = "None"
    if len((segmentation_array * brainstem_array).nonzero()[0]) >= t_wm:
        deep_wm = "Brainstem"

    if len((segmentation_array * corpus_callosum_array).nonzero()[0]) >= t_wm:
        deep_wm = "Corpus Callosum"

    if len((segmentation_array * internal_capsule_array).nonzero()[0]) >= t_wm:
        deep_wm = "Internal Capsule"
    deep_wm_f = np.nan
    if deep_wm == "None":
        deep_wm_f = 1
    if deep_wm != "None":
        deep_wm_f = 2

    if verbose:
        logger.debug("Determining cortical involvement")

    cortical_lesioned_voxels = len((segmentation_array * cortex_array).nonzero()[0])
    cortical_lesioned_voxels_f = np.nan
    if cortical_lesioned_voxels > cortical_thresh:
        cortical_lesioned_voxels_f = 2
    if cortical_lesioned_voxels <= cortical_thresh:
        cortical_lesioned_voxels_f = 1
    if verbose:
        logger.debug("Cortically lesioned voxels " + str(cortical_lesioned_voxels))

    # if verbose:
    #     logger.debug('Determining eloquent cortex involvement')

    # eloquent_lesioned_voxels = len((segmentation_array*eloquent_array).nonzero()[0])
    # eloquent_lesioned_voxels_f = np.nan
    # if eloquent_lesioned_voxels>eloquent_thresh:
    #     eloquent_lesioned_voxels_f=2
    # if eloquent_lesioned_voxels<=eloquent_thresh:
    #     eloquent_lesioned_voxels_f=1
    # if verbose:
    #     logger.debug(f'Eloquent volume: {eloquent_vol}')
    #     logger.debug('Eloquent cortex lesioned voxels '+str(eloquent_lesioned_voxels))
    if verbose:
        logger.debug("Determining eloquent cortex involvement")

    eloquent_labels = {
        2: "Speech motor",
        3: "Motor",
        4: "Vision",
    }

    eloquent_involved = []

    for code, lab in eloquent_labels.items():
        mask = eloquent_array == code
        eloquent_lesioned_voxels = np.count_nonzero(segmentation_array * mask)
        if eloquent_lesioned_voxels > eloquent_thresh:
            eloquent_involved.append(lab)
        if verbose:
            logger.debug(f"{lab} lesioned voxels: {eloquent_lesioned_voxels}")

    if verbose:
        logger.debug(f"Total eloquent volume: {eloquent_vol}")

    if len(eloquent_involved) == 0:
        eloquent_text = "No involvement"
    elif len(eloquent_involved) == 1:
        eloquent_text = eloquent_involved[0]
    else:
        eloquent_text = (
            " and ".join([", ".join(eloquent_involved[:-1]), eloquent_involved[-1]])
            if len(eloquent_involved) > 2
            else " and ".join(eloquent_involved)
        )

    if verbose:
        logger.debug(f"Eloquent regions involved: {eloquent_text}")

    if verbose:
        logger.debug("Determining midline involvement")

    if metadata is not None:
        oedema_cross_midline_f, nCET_cross_midline_f, CET_cross_midline_f = process_midline_stats(metadata=metadata)
    else:
        nCET_cross_midline = False
        nCET = np.asanyarray(segmentation.dataobj)
        nCET[nCET != nonenhancing_label] = 0
        nCET[nCET > 0] = 1
        temp = nCET.nonzero()[0]
        right_hemisphere = len(temp[temp < int(segmentation_array.shape[z_dim] / 2)])
        left_hemisphere = len(temp[temp > int(segmentation_array.shape[z_dim] / 2)])
        if right_hemisphere > midline_thresh and left_hemisphere > midline_thresh:
            nCET_cross_midline = True
        nCET_cross_midline_f = np.nan
        if nCET_cross_midline == True:
            nCET_cross_midline_f = 3
        if nCET_cross_midline == False:
            nCET_cross_midline_f = 2

        CET_cross_midline = False
        CET = np.asanyarray(segmentation.dataobj)
        CET[CET != enhancing_label] = 0
        CET[CET > 0] = 1
        temp = CET.nonzero()[0]
        right_hemisphere = len(temp[temp < int(segmentation_array.shape[z_dim] / 2)])
        left_hemisphere = len(temp[temp > int(segmentation_array.shape[z_dim] / 2)])
        if right_hemisphere > midline_thresh and left_hemisphere > midline_thresh:
            CET_cross_midline = True
        CET_cross_midline_f = np.nan
        if CET_cross_midline == True:
            CET_cross_midline_f = 3
        if CET_cross_midline == False:
            CET_cross_midline_f = 2

        oedema_cross_midline = False
        oedema = np.asanyarray(segmentation.dataobj)
        oedema[oedema != oedema_label] = 0
        oedema[oedema > 0] = 1
        temp = oedema.nonzero()[0]
        right_hemisphere = len(temp[temp < int(segmentation_array.shape[z_dim] / 2)])
        left_hemisphere = len(temp[temp > int(segmentation_array.shape[z_dim] / 2)])
        if right_hemisphere > midline_thresh and left_hemisphere > midline_thresh:
            oedema_cross_midline = True
        oedema_cross_midline_f = np.nan
        if oedema_cross_midline == True:
            oedema_cross_midline_f = 3
        if oedema_cross_midline == False:
            oedema_cross_midline_f = 2

    # if verbose:
    #     logger.debug('Deriving enhancing satellites')
    # labeled_array, num_components_cet = label(CET)
    # num_components_cet_f = np.nan
    # if num_components_cet>num_components_cet_thresh:
    #     num_components_cet_f =2
    # else:
    #     num_components_cet_f=1

    # if verbose:
    #     logger.debug('Deriving cysts')
    # labeled_array, num_components_ncet = label(nCET)
    # num_components_ncet_f =1
    # if num_components_ncet>cyst_thresh:
    #     num_components_ncet_f=2

    # if verbose:
    #     logger.debug('Cyst count '+str(num_components_ncet))

    # if verbose:
    #     logger.debug('Deriving enhancement thickness')

    # enhancing_skeleton = skeletonize(CET)
    # allpixels = np.count_nonzero(CET)
    # skeletonpixels = np.count_nonzero(enhancing_skeleton)

    # if allpixels>0:
    #     enhancing_thickness = allpixels/(skeletonpixels+1e-9)
    # if allpixels==0:
    #     enhancing_thickness=0
    # enhancing_thickness_f = np.nan
    # ll=3
    # if enhancing_thickness<ll:
    #     enhancing_thickness_f=3
    # if enhancing_thickness>=ll:
    #     enhancing_thickness_f=4
    # if enhancing_thickness>=ll and nonenhancing_voxels==0:
    #     enhancing_thickness_f=5

    # if verbose:
    #     logger.debug('Enhancing thickness: '+str(enhancing_thickness))

    CET_ss = np.asanyarray(segmentation_ss.dataobj)
    CET_ss[CET_ss != enhancing_label] = 0
    CET_ss[CET_ss > 0] = 1

    nCET_ss = np.asanyarray(segmentation_ss.dataobj)
    nCET_ss[nCET_ss != nonenhancing_label] = 0
    nCET_ss[nCET_ss > 0] = 1

    if verbose:
        logger.debug("Deriving enhancing satellites")
    labeled_array, num_components_cet = label(CET_ss)
    num_components_cet_f = np.nan
    if num_components_cet > num_components_cet_thresh:
        num_components_cet_f = 2
    else:
        num_components_cet_f = 1

    if verbose:
        logger.debug("Deriving cysts")
    labeled_array, num_components_ncet = label(nCET_ss)
    num_components_ncet_f = 1
    if num_components_ncet > cyst_thresh:
        num_components_ncet_f = 2

    if verbose:
        logger.debug("Cyst count " + str(num_components_ncet))

    if verbose:
        logger.debug("Deriving enhancement thickness")

    enhancing_skeleton = skeletonize(CET_ss)
    allpixels = np.count_nonzero(CET_ss)
    skeletonpixels = np.count_nonzero(enhancing_skeleton)

    if allpixels > 0:
        enhancing_thickness = allpixels / (skeletonpixels + 1e-9)
    if allpixels == 0:
        enhancing_thickness = 0
    enhancing_thickness_f = np.nan
    ll = 3
    if enhancing_thickness < ll:
        enhancing_thickness_f = 3
    if enhancing_thickness >= ll:
        enhancing_thickness_f = 4
    if enhancing_thickness >= ll and nonenhancing_voxels == 0:
        enhancing_thickness_f = 5

    if verbose:
        logger.debug("Enhancing thickness: " + str(enhancing_thickness))

    if verbose:
        logger.debug("Converting raw values to VASARI dictionary features")
    F1_dict = {
        "Frontal Lobe": 1,
        "Temporal Lobe": 2,
        "Insula": 3,
        "Parietal Lobe": 4,
        "Occipital Lobe": 5,
        "Brainstem": 6,
        "Corpus callosum": 7,
        "Thalamus": 8,
    }
    #  {1:"Frontal",2:"Temporal",3:"Insula",4:"Parietal",5:"Occipital",6:"Brainstem",7:"Corpus Callosum", 8:"Thalamus"}
    F2_dict = {"Right": 1, "Left": 3, "Bilateral": 2}

    proportion_enhancing_f = np.nan
    if proportion_enhancing <= 5:
        proportion_enhancing_f = 3
    if 5 < proportion_enhancing <= 33:
        proportion_enhancing_f = 4
    if 33 < proportion_enhancing <= 67:
        proportion_enhancing_f = 5
    if 67 < proportion_enhancing <= 100:
        proportion_enhancing_f = 6

    proportion_nonenhancing_f = np.nan
    if proportion_nonenhancing <= 5:
        proportion_nonenhancing_f = 3
    if 5 < proportion_nonenhancing <= 33:
        proportion_nonenhancing_f = 4
    if 33 < proportion_nonenhancing <= 67:
        proportion_nonenhancing_f = 5
    if 67 < proportion_nonenhancing <= 95:
        proportion_nonenhancing_f = 6
    if 95 < proportion_nonenhancing <= 99.5:
        proportion_nonenhancing_f = 7
    if proportion_nonenhancing > 99.5:  # allow for small segmentation variation
        proportion_nonenhancing_f = 8

    proportion_necrosis_f = np.nan
    if proportion_nonenhancing == 0:
        proportion_necrosis_f = 2
    if 0 < proportion_nonenhancing <= 5:
        proportion_necrosis_f = 3
    if 5 < proportion_nonenhancing <= 33:
        proportion_necrosis_f = 4
    if 33 < proportion_nonenhancing <= 67:
        proportion_necrosis_f = 5

    # Original multifocal logic
    # segmentation_array_binary = segmentation_array.copy()
    # segmentation_array_binary[segmentation_array_binary>0]=1
    # labeled_array_bin, num_components_bin = label(segmentation_array_binary)
    # f9_multifocal = 1
    # if num_components_bin>num_components_bin_thresh:
    #     f9_multifocal=2
    # if verbose:
    # logger.debug('Number of lesion components: '+str(num_components_bin))

    # New multifocal logic
    lesion_sizes = compute_ap_tv_cc_multifocal(
        file_ss,
        include_labels=[nonenhancing_label, enhancing_label],  # ncr+et
        cm_or_mm="cm",
        min_dim_thresh_cm=1.0,  # 0.5,
    )
    num_lesions = len(lesion_sizes)
    f9_multifocal = 1
    if num_lesions > 1:
        f9_multifocal = 2

    if verbose:
        logger.debug("Number of lesion components: " + str(num_lesions))

    proportion_oedema_f = np.nan
    if verbose:
        logger.debug("prop oedema " + str(proportion_oedema))

    if proportion_oedema == 0:
        proportion_oedema_f = 2
    if 0 < proportion_oedema <= 5:
        proportion_oedema_f = 3
    if 5 < proportion_oedema <= 33:
        proportion_oedema_f = 4
    if 33 < proportion_oedema:
        proportion_oedema_f = 5

    end_time = time.time()
    time_taken = end_time - start_time
    time_taken_round = np.round(time_taken, 2)
    if verbose:
        logger.debug("Time taken: " + str(time_taken_round) + " seconds")

    if verbose:
        logger.debug("")
        logger.debug("Complete! Generating output...")

    col_names = [
        "Tumor Location",
        "Side of Tumor Epicenter",
        "Eloquent Brain Involvement",
        "Enhancement Quality",
        "Proportion Enhancing",
        #    'Proportion nCET',
        "Proportion Necrosis",
        "Multifocal or Multicentric",
        "Thickness of enhancing margin",
        "Proportion of Oedema",
        "Ependymal (ventricular) Invasion",
        "Cortical involvement",
        "Deep WM invasion",
        # 'nCET Crosses Midline',
        "CET Crosses midline",
        "Multiple satellites present",
        "Region Proportions",
        "Lesion Sizes APxTVxCC (cm)"
    ]  # 'filename', 'reporter', 'time_taken_seconds', 'F8 Cyst(s)', 'F10 T1/FLAIR Ratio', 'F12 Definition of the Enhancing margin','F13 Definition of the non-enhancing tumour margin','F16 haemorrhage', 'F17 Diffusion','F18 Pial invasion', 'F25 Calvarial modelling', 'COMMENTS']

    if merged is not None:

        col_names.append("Asymmetrical Ventricles")
        col_names.append("Enlarged Ventricles")

        lvol, rvol = get_ventricle_volumes(merged)
        asymmetrical_ventricles = 0
        enlarged_ventricles = 0

        def is_compressed(v_affected, v_other, thresh=1.25):
            # If affected side ventricle is zero → extreme compression
            if v_affected == 0 and v_other > 0:
                return True
            # Avoid division-by-zero
            if v_other == 0:
                return False  # Cannot conclude compression
            return (v_other / v_affected) > thresh

        if side == "Left":
            if is_compressed(lvol, rvol):
                asymmetrical_ventricles = 1

        elif side == "Right":
            if is_compressed(rvol, lvol):
                asymmetrical_ventricles = 1

        else:  # Bilateral or unknown → fallback to general asymmetry
            if min(lvol, rvol) > 0:
                if max(lvol, rvol) / min(lvol, rvol) > 1.5:
                    asymmetrical_ventricles = 1
            else:
                if max(lvol, rvol) > 0:
                    asymmetrical_ventricles = 1

        if lvol > 20e3 or rvol > 20e3:
            enlarged_ventricles = 1

        if verbose:
            logger.debug(f"Left lateral ventricle volume:  {lvol}")
            logger.debug(f"Right lateral ventricle volume: {rvol}")
            logger.debug(f"Asymmetrical ventricles:        {asymmetrical_ventricles}")
            logger.debug(f"Enlarged ventricles:            {enlarged_ventricles}")
            logger.debug(f"Tumor side:                     {side}")

    result = pd.DataFrame(columns=col_names)
    result.loc[len(result)] = {
        #   'filename':file,
        #    'reporter':'VASARI-auto',
        #   'time_taken_seconds':time_taken_round,
        "Tumor Location": F1_dict[vols.iloc[0, 0]],  # vols.iloc[0,0],
        "Side of Tumor Epicenter": F2_dict[side],
        "Eloquent Brain Involvement": eloquent_text,  # np.nan, #unsupported in current version
        "Enhancement Quality": enhancement_quality,
        "Proportion Enhancing": proportion_enhancing_f,
        #   'Proportion nCET':proportion_nonenhancing_f, # brats labels don't include ncet
        "Proportion Necrosis": proportion_necrosis_f,
        #   'F8 Cyst(s)':np.nan, #unsupported in current version num_components_ncet_f,
        "Multifocal or Multicentric": f9_multifocal,
        #    'F10 T1/FLAIR Ratio':np.nan,  #unsupported in current version
        "Thickness of enhancing margin": enhancing_thickness_f,
        #    'F12 Definition of the Enhancing margin':np.nan,  #unsupported in current version
        #    'F13 Definition of the non-enhancing tumour margin':np.nan,  #unsupported in current version
        "Proportion of Oedema": proportion_oedema_f,
        "Edema crosses midline": oedema_cross_midline_f,
        #    'F16 haemorrhage':np.nan,  #unsupported in current version
        #    'F17 Diffusion':np.nan,  #unsupported in current version
        #    'F18 Pial invasion':np.nan, #unsupported in current version
        "Ependymal (ventricular) Invasion": ependymal,
        "Cortical involvement": cortical_lesioned_voxels_f,
        "Deep WM invasion": deep_wm_f,
        #    'nCET Crosses Midline':nCET_cross_midline_f,# brats labels don't include ncet
        "CET Crosses midline": CET_cross_midline_f,
        "Multiple satellites present": num_components_cet_f,
        #    'F25 Calvarial modelling':np.nan, #unsupported in current version
        "Asymmetrical Ventricles": asymmetrical_ventricles,
        "Enlarged Ventricles": enlarged_ventricles,
        "Region Proportions": region_prop_list,
        "Lesion Sizes APxTVxCC (cm)": lesion_sizes,
    }
    return result


def compute_ap_tv_cc_multifocal(
    mask_path, include_labels=[1, 2], cm_or_mm="mm", min_dim_thresh_cm=3.0, debug_save_path=None, voxel_sizes=[1.0, 1.0, 1.0]
):
    """
    Compute AP x TV x CC dimensions for each lesion component.
    Keeps the largest lesion regardless of threshold.
    Remaining lesions are filtered by minimum dimension threshold.
    """
    img = nib.load(mask_path)
    data = np.asanyarray(img.dataobj)
    affine = img.affine

    # Restrict to selected labels
    mask = np.isin(data, include_labels)
    if not np.any(mask):
        if debug_save_path:
            nib.save(nib.Nifti1Image(np.zeros_like(mask, dtype=np.uint8), affine), debug_save_path)
        return []

    # Label connected components
    labeled, num = label(mask)

    if voxel_sizes is None:
        voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    else:
        voxel_sizes = np.asarray(voxel_sizes, dtype=float)

    # Precompute lesion volumes to find largest
    lesion_volumes = {}
    for lesion_id in range(1, num + 1):
        lesion_voxels = np.sum(labeled == lesion_id)
        lesion_volumes[lesion_id] = lesion_voxels

    # ID of largest lesion
    largest_lesion_id = max(lesion_volumes, key=lesion_volumes.get)

    results = []
    selected_mask = np.zeros_like(labeled, dtype=np.uint16) if debug_save_path else None
    next_label = 1

    for lesion_id in range(1, num + 1):
        lesion_mask = labeled == lesion_id
        coords = np.argwhere(lesion_mask)
        if coords.size == 0:
            continue

        # Bounding box (voxel extents)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        size_voxels = np.array([x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1])

        # Convert to mm
        size_mm = size_voxels * voxel_sizes[[0, 1, 2]]
        TV_mm, AP_mm, CC_mm = size_mm

        # Convert units
        if cm_or_mm == "cm":
            AP, TV, CC = AP_mm / 10.0, TV_mm / 10.0, CC_mm / 10.0
        else:
            AP, TV, CC = AP_mm, TV_mm, CC_mm

        # Determine if lesion passes threshold
        if cm_or_mm == "mm":
            threshold_pass = all(dim >= min_dim_thresh_cm * 10.0 for dim in (AP, TV, CC))
        else:
            threshold_pass = all(dim >= min_dim_thresh_cm for dim in (AP, TV, CC))

        # Keep largest lesion always
        if lesion_id == largest_lesion_id or threshold_pass:
            results.append((AP, TV, CC))

            if debug_save_path:
                selected_mask[lesion_mask] = next_label
                next_label += 1

    # Save debug mask
    if debug_save_path:
        nib.save(nib.Nifti1Image(selected_mask, affine), debug_save_path)

    return results


def get_ventricle_volumes(merged_mask_path, spacing_mm=(1.0, 1.0, 1.0)):
    spacing_scalar = math.prod(spacing_mm)
    LEFT_LATERAL_VENTRICLE_IDX, RIGHT_LATERAL_VENTRICLE_IDX = 4, 43
    ss = nib.load(merged_mask_path).get_fdata()

    vol_left_mm = np.sum(ss == LEFT_LATERAL_VENTRICLE_IDX) * spacing_scalar
    vol_right_mm = np.sum(ss == RIGHT_LATERAL_VENTRICLE_IDX) * spacing_scalar

    return vol_left_mm, vol_right_mm


def get_vasculature(
    t1c_path,
    save_path,
    window=[3000, 15000],
    tumor_path=None,
):
    t1c_im = nib.load(t1c_path)
    t1c = t1c_im.get_fdata()
    windowed = np.where((t1c >= window[0]) & (t1c <= window[1]), t1c, 0)

    if tumor_path is not None:
        tumor = nib.load(tumor_path).get_fdata()
        enh = np.where(tumor == 3, 5, 0)
        windowed = np.where(enh == 5, 0, windowed)

    nib.save(nib.Nifti1Image(windowed, affine=t1c_im.affine, header=t1c_im.header), save_path)
    logger.debug(f'Processed {t1c_path.split("/")[-1]}, saved vessel map to {save_path}')
