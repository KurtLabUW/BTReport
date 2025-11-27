# This is the codebase for VASARI-auto, modified by Juampablo Heras Rivera for BTReport
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
#   - Refactored as ExtractVASARI class
#   - Improved midline crossing logic in the original subject space
#   - Multifocal or Multicentric logic: used connected components to get the number of tumor bodies, then only kept mutliple lesions if the non-largest lesions are >1cm.  Lesion is multifocal when more than one of these lesions exists
#   - Used the images image in their original space instead of the version registered MNI152 space to better estimate quantities like proportion necrotic, etc.. This was noted as an option in the original paper, but they used the MNI152 version.
#   - Added eloquent regions obtained from Brodmann Area Maps (https://surfer.nmr.mgh.harvard.edu/fswiki/BrodmannAreaMaps)
#   - Added lesion sizes APxTVxCC
#   - Adjusted to only include quantities that can be derived from BraTS masks. Removed proportion nCET e.g.
#   - Changed proportions to continuous floats
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
import attrs
from os.path import join
pd.set_option("display.max_rows", 500)
import math

from pprint import pprint
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@attrs.define
class NiftiImage:
    path: str
    spacing: tuple[float, float, float] = None   
    img: nib.Nifti1Image = attrs.field(init=False)
    array: np.ndarray = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.img = nib.load(self.path)
        self.array = np.asanyarray(self.img.dataobj)
        if self.spacing is None:
            self.spacing = self.img.header.get_zooms()[:3]

    @property
    def volume(self) -> float:
        """binarized mask volume returned in mL"""
        dx, dy, dz = self.spacing
        voxel_mm3 = dx * dy * dz
        return float(np.count_nonzero(self.array) * voxel_mm3 / 1000.0)

@attrs.define
class ExtractVASARI:
    # tumorseg_mni: str
    # tumorseg_ss: str | None = None
    # merged: str | None = None
    # metadata: dict | None = None
    verbose: bool = False

    atlases: str = "btreport/vasari_features/atlas_masks"
    # label values
    enhancing_label: int = 4
    nonenhancing_label: int = 1
    oedema_label: int = 2

    # geometry + scaling params
    z_dim: int = -1
    cf: float = 1.0
    resolution: float = 1.0

    # thresholds
    t_ependymal: int = 5000
    t_wm: int = 100
    midline_thresh: int = 5
    enh_quality_thresh: int = 15
    cyst_thresh: int = 50
    cortical_thresh: int = 1000
    eloquent_thresh: int = 100
    focus_thresh: int = 30000

    # connected component thresholds
    num_components_bin_thresh: int = 10
    num_components_cet_thresh: int = 15

    atlas_regions = [
    "brainstem", "midline", "frontal_lobe", "insula", "occipital",
    "parietal", "temporal", "thalamus", "corpus_callosum",
    "ventricles", "internal_capsule", "cortex", "eloquent_grouped"]

    COL_NAMES = [
        "Tumor Location",
        "Side of Tumor Epicenter",
        "Eloquent Brain Involvement",
        "Enhancement Quality",
        "Proportion Enhancing",
        "Proportion Necrosis",
        "Multifocal or Multicentric",
        "Thickness of enhancing margin",
        "Proportion of Oedema",
        "Edema crosses midline",
        "Ependymal (ventricular) Invasion",
        "Cortical involvement",
        "Deep WM invasion",
        "CET Crosses midline",
        "Multiple satellites present",
        "Asymmetrical Ventricles",
        "Enlarged Ventricles",
        "Region Proportions",
        "Lesion Sizes APxTVxCC (cm)",
        "NCR Volume (mL)",
        "ED Volume (mL)",
        "ET Volume (mL)",
        "Total tumor volume (mL)",
        "Number of lesions",
        "Asymmetrical Ventricles",
        "Enlarged Ventricles",
    ]

    def get_laterality(self, segmentation_array):
        temp = segmentation_array.nonzero()[0]
        mid = segmentation_array.shape[self.z_dim] // 2
        right_hemi = (temp < mid).sum()
        left_hemi  = (temp > mid).sum()
        if right_hemi > left_hemi:   side = "Right"
        if left_hemi  > right_hemi:  side = "Left"
        if right_hemi > self.focus_thresh and left_hemi > self.focus_thresh:
            side = "Bilateral"
        return side
    
    def get_region_proportions(self, segmentation, atlas_regions):
        segmentation_array = segmentation.array.copy()
        segmentation_array[segmentation_array == self.oedema_label] = 0
        segmentation_array[segmentation_array == self.enhancing_label] = 1
        segmentation_array[segmentation_array == self.nonenhancing_label] = 1
        if segmentation_array.sum() == 0:
            if self.verbose: logger.debug("No lesion detected, falling back to oedema label for closer inspection")
            segmentation_array = segmentation.array
            segmentation_array[segmentation_array != self.oedema_label] = 0
        
        proportions = {
            region_name: 
            (segmentation_array * atlas_regions[region_name].array).nonzero()[0].size / (segmentation_array.sum() + self.cf) for region_name in atlas_regions.keys()
        }

        d = {"ROI": list(proportions.keys()), "prop": list(proportions.values())}
        vols = pd.DataFrame(d).sort_values(by="prop", ascending=False).reset_index(drop=True)
        region_prop_list = [(r.ROI, r.prop) for _, r in vols.iterrows() if r.prop >= 0.1]
        if self.verbose:
            logger.debug(f"Regions with >=10% tumor involvement: {region_prop_list}")

        ## ORIGINAL LOGIC
        BILOBAR_FULL = {
            ("Frontal Lobe","Parietal Lobe"):   "Fronto-Parietal",
            ("Parietal Lobe","Frontal Lobe"):   "Parieto-Frontal",

            ("Frontal Lobe","Temporal Lobe"):   "Fronto-Temporal",
            ("Temporal Lobe","Frontal Lobe"):   "Temporo-Frontal",

            ("Frontal Lobe","Insula"):          "Fronto-Insular",
            ("Insula","Frontal Lobe"):          "Insulo-Frontal",

            ("Frontal Lobe","Occipital Lobe"):  "Fronto-Occipital",
            ("Occipital Lobe","Frontal Lobe"):  "Occipito-Frontal",

            ("Temporal Lobe","Parietal Lobe"):  "Temporo-Parietal",
            ("Parietal Lobe","Temporal Lobe"):  "Parieto-Temporal",

            ("Temporal Lobe","Occipital Lobe"): "Temporo-Occipital",
            ("Occipital Lobe","Temporal Lobe"): "Occipito-Temporal",

            ("Temporal Lobe","Insula"):         "Temporo-Insular",
            ("Insula","Temporal Lobe"):         "Insulo-Temporal",

            ("Parietal Lobe","Occipital Lobe"): "Parieto-Occipital",
            ("Occipital Lobe","Parietal Lobe"): "Occipito-Parietal",

            ("Parietal Lobe","Insula"):         "Parieto-Insular",
            ("Insula","Parietal Lobe"):         "Insulo-Parietal",
        }
        regions = [name for name,_ in region_prop_list]     

        if len(regions)==1:
            tumor_location = regions[0]

        elif len(regions)==2:
            if tuple(regions) in BILOBAR_FULL:
                tumor_location = BILOBAR_FULL[tuple(regions)] + " region"
            else:
                r1 = regions[0].replace(" Lobe","")
                r2 = regions[1].replace(" Lobe","")
                tumor_location = f"{r1}-{r2} region"

        else:
            # 3+ lobes
            parts = [r.replace(" Lobe","") for r in regions]
            tumor_location = f"{', '.join(parts[:-1])} and {parts[-1]} lobes"

        return proportions, region_prop_list, vols, tumor_location


    def get_eloquent_involvement(self, segmentation, atlas_regions):
        """
        Determine eloquent cortex involvement using segmented tumor + atlas annotations.
        """
        eloquent_labels = {2: "Speech motor", 3: "Motor", 4: "Vision",}

        eloquent_involved = []
        for code, label in eloquent_labels.items():
            mask = (atlas_regions['eloquent_grouped'].array == code)
            lesioned_voxels = np.count_nonzero(segmentation.array * mask)

            if lesioned_voxels > self.eloquent_thresh:
                eloquent_involved.append(label)

            if getattr(self, "verbose", False):
                logger.debug(f"{label} lesioned voxels = {lesioned_voxels}")

        if len(eloquent_involved) == 0:
            return "No involvement"
        if len(eloquent_involved) == 1:
            return eloquent_involved[0]
        else:
            return (
                " and ".join([", ".join(eloquent_involved[:-1]), eloquent_involved[-1]])
                if len(eloquent_involved) > 2
                else " and ".join(eloquent_involved)
            )


    def process_midline_stats(self, metadata):
        bool_to_index = {True: 3, False: 2}
        oedema_cross_midline_f = bool_to_index[metadata["crosses_ideal_midline"]["ed"]]
        nCET_cross_midline_f = bool_to_index[metadata["crosses_ideal_midline"]["ed"] or metadata["crosses_ideal_midline"]["ncr"]]
        CET_cross_midline_f = bool_to_index[metadata["crosses_ideal_midline"]["et"]]
        return oedema_cross_midline_f, nCET_cross_midline_f, CET_cross_midline_f

    def get_midline_involvement(self, metadata=None, segmentation=None):
        assert (metadata is not None) ^ (segmentation is not None), 'Exactly one of metadata or segmentation should be provided.'
        if metadata is not None:
            oedema_cross_midline_f, nCET_cross_midline_f, CET_cross_midline_f = self.process_midline_stats(metadata)
        else:
            nCET_cross_midline = False
            nCET = segmentation.array.copy()
            nCET[nCET != self.nonenhancing_label] = 0
            nCET[nCET > 0] = 1
            temp = nCET.nonzero()[0]
            right_hemisphere = len(temp[temp < int(segmentation.array.shape[self.z_dim] / 2)])
            left_hemisphere = len(temp[temp > int(segmentation.array.shape[self.z_dim] / 2)])
            if right_hemisphere > self.midline_thresh and left_hemisphere > self.midline_thresh:
                nCET_cross_midline = True
            nCET_cross_midline_f = np.nan
            if nCET_cross_midline == True:
                nCET_cross_midline_f = 3
            if nCET_cross_midline == False:
                nCET_cross_midline_f = 2

            CET_cross_midline = False
            CET = segmentation.array.copy()
            CET[CET != self.enhancing_label] = 0
            CET[CET > 0] = 1
            temp = CET.nonzero()[0]
            right_hemisphere = len(temp[temp < int(segmentation.array.shape[self.z_dim] / 2)])
            left_hemisphere = len(temp[temp > int(segmentation.array.shape[self.z_dim] / 2)])
            if right_hemisphere > self.midline_thresh and left_hemisphere > self.midline_thresh:
                CET_cross_midline = True
            CET_cross_midline_f = np.nan
            if CET_cross_midline == True:
                CET_cross_midline_f = 3
            if CET_cross_midline == False:
                CET_cross_midline_f = 2

            oedema_cross_midline = False
            oedema = segmentation.array.copy()
            oedema[oedema != self.oedema_label] = 0
            oedema[oedema > 0] = 1
            temp = oedema.nonzero()[0]
            right_hemisphere = len(temp[temp < int(segmentation.array.shape[self.z_dim] / 2)])
            left_hemisphere = len(temp[temp > int(segmentation.array.shape[self.z_dim] / 2)])
            if right_hemisphere > self.midline_thresh and left_hemisphere > self.midline_thresh:
                oedema_cross_midline = True
            oedema_cross_midline_f = np.nan
            if oedema_cross_midline == True:
                oedema_cross_midline_f = 3
            if oedema_cross_midline == False:
                oedema_cross_midline_f = 2      
        return oedema_cross_midline_f, nCET_cross_midline_f, CET_cross_midline_f

    def get_presence_of_cysts_and_satellites(self, segmentation_array):
        CET_ss = segmentation_array.copy()
        CET_ss[CET_ss != self.enhancing_label] = 0
        CET_ss[CET_ss > 0] = 1

        nCET_ss = segmentation_array.copy()
        nCET_ss[nCET_ss != self.nonenhancing_label] = 0
        nCET_ss[nCET_ss > 0] = 1

        _, num_components_cet = label(CET_ss)
        _, num_components_ncet = label(nCET_ss)

        num_components_cet_f = 2 if num_components_cet > self.num_components_cet_thresh else 1
        num_components_ncet_f = 2 if num_components_ncet > self.cyst_thresh else 1

        if self.verbose: logger.debug("Cyst count " + str(num_components_ncet))
        if self.verbose: logger.debug("Enhancing satetllite count " + str(num_components_cet))

        return num_components_cet_f, num_components_ncet_f

    def get_enhancing_thickness(self, segmentation_array):
        nonenhancing_voxels = np.count_nonzero(segmentation_array == self.nonenhancing_label)
        voxel_length_mm = np.cbrt(self.resolution) # approximation of 1D length for voxels
        CET_ss = segmentation_array.copy()
        CET_ss[CET_ss != self.enhancing_label] = 0
        CET_ss[CET_ss > 0] = 1
        enhancing_skeleton = skeletonize(CET_ss)
        allpixels = np.count_nonzero(CET_ss)
        skeletonpixels = np.count_nonzero(enhancing_skeleton)
        if allpixels > 0:
            enhancing_thickness = allpixels / (skeletonpixels * voxel_length_mm + 1e-9)
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
        if self.verbose: logger.debug(f"Enhancing thickness = {enhancing_thickness:.3f} mm")
        return enhancing_thickness_f

    def compute_ap_tv_cc_multifocal(self, segmentation, include_labels=[1, 2], cm_or_mm="mm", min_dim_thresh_cm=3.0, debug_save_path=None):
        """
        Compute AP x TV x CC dimensions for each lesion component.
        Keeps the largest lesion regardless of threshold.
        Remaining lesions are filtered by minimum dimension threshold.
        """
        data = segmentation.array
        affine = segmentation.img.affine

        # Restrict to selected labels
        mask = np.isin(data, include_labels)
        if not np.any(mask):
            if debug_save_path:
                nib.save(nib.Nifti1Image(np.zeros_like(mask, dtype=np.uint8), affine), debug_save_path)
            return []

        # Label connected components
        labeled, num = label(mask)

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
            size_mm = size_voxels * self.resolution
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

    def get_multifocal_or_multicentric(self, segmentation):
        lesion_sizes = self.compute_ap_tv_cc_multifocal(segmentation, 
                                                        include_labels=[self.nonenhancing_label, self.enhancing_label],  # ncr+et
                                                        cm_or_mm="cm",
                                                        min_dim_thresh_cm=0.5, #1.0,  # 0.5,
                                                        )      
        num_lesions = len(lesion_sizes)
        f9_multifocal = 2 if num_lesions > 1 else 1
        if self.verbose: logger.debug("Number of lesion components: " + str(num_lesions))
        return lesion_sizes, num_lesions, f9_multifocal


    def get_ventricle_volumes(self, merged_anatseg_array):
        LEFT_LATERAL_VENTRICLE_IDX, RIGHT_LATERAL_VENTRICLE_IDX = 4, 43 ### HARDCODED from synthseg
        vol_left_mm = np.sum(merged_anatseg_array == LEFT_LATERAL_VENTRICLE_IDX) * self.resolution
        vol_right_mm = np.sum(merged_anatseg_array == RIGHT_LATERAL_VENTRICLE_IDX) * self.resolution
        return vol_left_mm, vol_right_mm


    def get_ventricle_geometry_statistics(self, merged_anatseg_array, side, thresh_asym=1.25, enlarge_thresh=20e3):
        """
        Compute L/R ventricle volume (mL), asymmetry flag, enlargement flag, and side dominance.
        merged_anatseg : labeled anatomical+tumor segmentation (same format as input to get_ventricle_volumes)
        """

        lvol, rvol = self.get_ventricle_volumes(merged_anatseg_array)   

        def is_compressed(v_affected, v_other, thresh=thresh_asym):
            if v_affected == 0 and v_other > 0:
                return True
            if v_other == 0:
                return False
            return (v_other / v_affected) > thresh

        asymmetrical_ventricles = 0

        if side.lower() == "left":
            if is_compressed(lvol, rvol):
                asymmetrical_ventricles = 1

        elif side.lower() == "right":
            if is_compressed(rvol, lvol):
                asymmetrical_ventricles = 1

        else:   
            if min(lvol, rvol) > 0:
                if max(lvol, rvol) / min(lvol, rvol) > 1.5:
                    asymmetrical_ventricles = 1
            else:
                if max(lvol, rvol) > 0:
                    asymmetrical_ventricles = 1

        enlarged_ventricles = 1 if (lvol > enlarge_thresh or rvol > enlarge_thresh) else 0

        if getattr(self, "verbose", False):
            logger.debug(f"Left ventricular volume:     {lvol}")
            logger.debug(f"Right ventricular volume:    {rvol}")
            logger.debug(f"Asymmetrical ventricles:     {asymmetrical_ventricles}")
            logger.debug(f"Enlarged ventricles:         {enlarged_ventricles}")
            logger.debug(f"Tumor side:       {side}")

        return lvol, rvol, asymmetrical_ventricles, enlarged_ventricles



    def __call__(self, tumorseg_mni, tumorseg_ss, merged=None, metadata=None):
        """Run full VASARI feature extraction"""
        start_time = time.time()
        atlas_regions = {k:NiftiImage(join(self.atlases, f'{k}.nii.gz')) for k in self.atlas_regions}
        segmentation = NiftiImage(tumorseg_mni)
        segmentation_ss = NiftiImage(tumorseg_ss)
        if merged:
            merged_anat_tumor = NiftiImage(merged)

        if self.verbose: logger.debug("Running voxel quantification per tissue class")
        total_lesion_burden = np.count_nonzero(segmentation_ss.array)
        enhancing_voxels    = np.count_nonzero(segmentation_ss.array == self.enhancing_label)
        nonenhancing_voxels = np.count_nonzero(segmentation_ss.array == self.nonenhancing_label)
        oedema_voxels       = np.count_nonzero(segmentation_ss.array == self.oedema_label)
        proportion_enhancing = np.round( (enhancing_voxels / (total_lesion_burden + 0.1)) * 100 , 2)
        proportion_nonenhancing = np.round( (nonenhancing_voxels / (total_lesion_burden + 0.1)) * 100 , 2)
        proportion_oedema = np.round( ((oedema_voxels / (total_lesion_burden + 0.1))) * 100, 2)
        ncr_vol_ml = nonenhancing_voxels * self.resolution / 1000.0
        ed_vol_ml  = oedema_voxels  * self.resolution / 1000.0
        et_vol_ml  = enhancing_voxels  * self.resolution / 1000.0
        global_vol_ml = total_lesion_burden * self.resolution / 1000.0

        if self.verbose: logger.debug("Proportion Edema " + str(proportion_oedema))  
        if self.verbose: logger.debug("Proportion Enhancing " + str(proportion_enhancing))  
        if self.verbose: logger.debug("Proportion NCR " + str(proportion_nonenhancing))  

        
        if self.verbose: logger.debug("Deriving number of components")
        labeled_array, num_components = label(segmentation_ss.array)


        if self.verbose: logger.debug("Determining laterality")
        side = self.get_laterality(segmentation_array=segmentation.array)


        if self.verbose: logger.debug("Determining proportions")
        proportions, region_prop_list, vols, tumor_location = self.get_region_proportions(segmentation, atlas_regions)
        

        if self.verbose: logger.debug("Determining enhancement quality (f4)")
        enhancement_quality = 1
        if proportion_enhancing > 0:
            if proportion_enhancing > self.enh_quality_thresh:
                enhancement_quality = 3
            else:
                enhancement_quality = 2


        if self.verbose: logger.debug("Determining ependymal extension (f19)")
        if len((segmentation.array * atlas_regions['ventricles'].array).nonzero()[0]) >= self.t_ependymal:
            ependymal = 2
        else:
            ependymal = 1


        if self.verbose: logger.debug("Determining deep white matter involvement (f21)")
        deep_wm_invaded = []
        if len((segmentation.array * atlas_regions['brainstem'].array).nonzero()[0]) >= self.t_wm: deep_wm_invaded.append("Brainstem")
        if len((segmentation.array * atlas_regions['corpus_callosum'].array).nonzero()[0]) >= self.t_wm: deep_wm_invaded.append("Corpus Callosum")
        if len((segmentation.array * atlas_regions['internal_capsule'].array).nonzero()[0])>= self.t_wm: deep_wm_invaded.append("Internal Capsule")
        deep_wm_f = 1 if len(deep_wm_invaded) == 0 else 2  
        if self.verbose and deep_wm_f ==2: logger.debug(f"Deep WM regions invaded: {deep_wm_invaded}")

        
        if self.verbose: logger.debug("Determining cortical involvement (f20)")
        cortical_lesioned_voxels   = len((segmentation.array * atlas_regions['cortex'].array).nonzero()[0])
        cortical_lesioned_voxels_f = 2 if cortical_lesioned_voxels > self.cortical_thresh else 1
        if self.verbose: logger.debug(f"Cortically lesioned voxels: {cortical_lesioned_voxels}")


        if self.verbose: logger.debug("Determining eloquent cortex involvement (f3) using Brodmann Area Maps Atlas")
        eloquent_text = self.get_eloquent_involvement(segmentation, atlas_regions)

        if self.verbose: logger.debug("Determining midline involvement (f22, f23)")
        oedema_cross_midline_f, nCET_cross_midline_f, CET_cross_midline_f = self.get_midline_involvement(metadata=metadata, segmentation=None if metadata else segmentation)
      

        if self.verbose: logger.debug("Determining presence of cysts (f8) and enhancing satellites (f24)")
        num_components_cet_f, num_components_ncet_f= self.get_presence_of_cysts_and_satellites(segmentation_ss.array)

        if self.verbose: logger.debug("Determining enhacement thickness (f11)")
        enhancing_thickness_f = self.get_enhancing_thickness(segmentation_ss.array)


        if self.verbose: logger.debug("Determining multifocality or multicentricity (f9)")
        lesion_sizes, num_lesions, f9_multifocal = self.get_multifocal_or_multicentric(segmentation_ss)

        if merged_anat_tumor:
            if self.verbose: logger.debug("Determining ventricle geometry anomalies")
            lvol, rvol, asymmetrical_ventricles, enlarged_ventricles = self.get_ventricle_geometry_statistics(merged_anatseg_array=merged_anat_tumor.array, side=side)

        

        if self.verbose: logger.debug("Converting raw values to VASARI dictionary features")
        result = pd.DataFrame(columns=self.COL_NAMES)
        result.loc[len(result)] = {
            #   'filename':file,
            #    'reporter':'VASARI-auto',
            #   'time_taken_seconds':time_taken_round,
            "Tumor Location": tumor_location,  # vols.iloc[0,0],
            "Side of Tumor Epicenter": side,
            "Eloquent Brain Involvement": eloquent_text,  # np.nan, #unsupported in current version
            "Enhancement Quality": enhancement_quality,
            "Proportion Enhancing": proportion_enhancing,
            #   'Proportion nCET':proportion_nonenhancing_f, # brats labels don't include ncet
            "Proportion Necrosis": proportion_nonenhancing,
            #   'F8 Cyst(s)':np.nan, #unsupported in current version num_components_ncet_f,
            "Multifocal or Multicentric": f9_multifocal,
            #    'F10 T1/FLAIR Ratio':np.nan,  #unsupported in current version
            "Thickness of enhancing margin": enhancing_thickness_f,
            #    'F12 Definition of the Enhancing margin':np.nan,  #unsupported in current version
            #    'F13 Definition of the non-enhancing tumour margin':np.nan,  #unsupported in current version
            "Proportion of Oedema": proportion_oedema,
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
            "NCR Volume (mL)":float(ncr_vol_ml),
            "ED Volume (mL)": float(ed_vol_ml),
            "ET Volume (mL)": float(et_vol_ml),
            "Total tumor volume (mL)":float(global_vol_ml),
            "Number of lesions":len(lesion_sizes), 
        }

        end_time = time.time()
        time_taken = np.round(end_time - start_time, 2)
        if self.verbose: logger.debug("Time taken: " + str(time_taken) + " seconds")

        return result

if __name__ == '__main__':
    tumorseg_mni='/pscratch/sd/j/jehr/MSFT/BTReport/data/example/45203724572086/tmp/tumor_seg_in_MNI152_space.nii.gz'
    tumorseg_ss = '/pscratch/sd/j/jehr/MSFT/BTReport/data/example/45203724572086/45203724572086-seg.nii.gz'
    merged = '/pscratch/sd/j/jehr/MSFT/BTReport/data/example/45203724572086/tmp/MNI152_in_subject_space_merged_seg.nii.gz'
    metadata = {
        "crosses_ideal_midline": {
            "ncr": True,
            "ed": False,
            "et": True,
            "ncr_et": True,   
        }
    }
    extractor = ExtractVASARI(verbose=False)
    result = extractor(tumorseg_mni, tumorseg_ss, merged, metadata)

    pprint(result.iloc[0].to_dict())



                
    ######## TODO: add ventricular effacement! add synthseg regions involved