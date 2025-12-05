import pandas as pd

"""
VASARI_FEATURES = [
    "Tumor Location",
    "Side of Tumor Epicenter",
    "Eloquent Brain Involvement",
    "Enhancement Quality",
    "Proportion Enhancing",
    "Proportion nCET",
    'Proportion Necrosis',#
    "Multifocal or Multicentric",
    "Thickness of enhancing margin",
    "Proportion of Oedema",
    "Edema crosses midline",#
    "Ependymal (ventricular) Invasion",
    "Cortical involvement",
    "Deep WM invasion",
    "nCET Crosses Midline",#
    "CET Crosses midline",#
    "Multiple satellites present",
    "Asymmetrical Ventricles",
    "Enlarged Ventricles", 
]
"""


VASARI_MAPS = {
    "Tumor Location": {
        1: "Frontal",
        2: "Temporal",
        3: "Insula",
        4: "Parietal",
        5: "Occipital",
        6: "Brainstem",
        7: "Corpus Callosum",
        8: "Thalamus",
    },  # {1:"Frontal",2:"Temporal",3:"Insula",4:"Parietal",5:"Occipital",6:"Brainstem",8:"Corpus Callosum"},
    "Side of Tumor Epicenter": {1: "Right", 2: "Bilateral", 3: "Left"},
    "Eloquent Brain Involvement": {1: "No involvement", 2: "Speech motor", 3: "Motor", 4: "Vision"},
    "Enhancement Quality": {1: "None", 2: "Mild", 3: "Marked"},
    # "Proportion Enhancing": {3: "<5%", 4: "6-33%", 5: "34-67%", 6: ">68%"},
    # "Proportion nCET": {3: "<5%", 4: "6-33%", 5: "34-67%", 6: "68-95%", 7: "95-99%", 8: "100%"},
    # "Proportion Necrosis": {2: "0%", 3: "<5%", 4: "6-33%", 5: "34-67%"},  #
    "Multifocal or Multicentric": {1: "Solitary", 2: "Multifocal", 3: "Multicentric", 4: "Gliomatosis"},
    "Thickness of enhancing margin": {3: "<3mm", 4: ">3mm", 5: "Solid"},
    # "Proportion of Oedema": {2: "0%", 3: "<5%", 4: "6-33%", 5: "34-67%"},
    "Edema crosses midline": {3: "True", 2: "False"},  #
    "Ependymal (ventricular) Invasion": {1: "Absent", 2: "Present"},
    "Cortical involvement": {1: "Absent", 2: "Present"},
    "Deep WM invasion": {1: "Absent", 2: "Present"},
    "nCET Crosses Midline": {3: "True", 2: "False"},  #
    "CET Crosses midline": {3: "True", 2: "False"},  #
    "Multiple satellites present": {1: "Absent", 2: "Present"},
    "Asymmetrical Ventricles": {0: "Absent", 1: "Present"},
    "Enlarged Ventricles": {0: "Absent", 1: "Present"},
}

VASARI_SENTENCE_MAPS = {
    "Tumor Location": {
        1: "Tumor centered in the frontal lobe.",
        2: "Tumor centered in the temporal lobe.",
        3: "Tumor centered in the insula.",
        4: "Tumor centered in the parietal lobe.",
        5: "Tumor centered in the occipital lobe.",
        6: "Tumor centered in the brainstem.",
        7: "Tumor centered in the corpus callosum.",
        8: "Tumor centered in the thalamus.",
    },
    "Side of Tumor Epicenter": {
        1: "Tumor predominantly affects the right hemisphere.",
        2: "Tumor spans both hemispheres, indicating bilateral involvement.",
        3: "Tumor predominantly affects the left hemisphere.",
    },
    "Eloquent Brain Involvement": {
        1: "No involvement of eloquent cortical regions.",
        2: "Tumor involves the speech motor cortex.",
        3: "Tumor involves motor cortex regions.",
        4: "Tumor involves visual cortex regions.",
    },
    "Enhancement Quality": {1: "No contrast enhancement is observed.", 2: "Mild enhancement is noted.", 3: "Marked enhancement is noted."},
    "Proportion Enhancing": {
        3: "Enhancing component constitutes less than 5% of the lesion.",
        4: "Enhancing component comprises roughly 6-33% of the lesion.",
        5: "Enhancing component involves approximately 34-67% of the lesion volume.",
        6: "Enhancing tissue represents more than 68% of the lesion.",
    },
    "Proportion nCET": {
        3: "Non-enhancing tumor component is minimal (<5%).",
        4: "Non-enhancing tumor component comprises roughly 6-33% of the lesion.",
        5: "Non-enhancing tumor component involves approximately 34-67% of the lesion.",
        6: "Non-enhancing tumor component involves 68-95% of the lesion.",
        7: "Non-enhancing tumor component occupies 95-99% of the lesion volume.",
        8: "Lesion is entirely non-enhancing.",
    },
    "Proportion Necrosis": {
        2: "No visible necrosis within the enhancing lesion.",
        3: "Minimal central necrosis (<5%) is present.",
        4: "Moderate necrosis (6-33%) is identified within the tumor core.",
        5: "Extensive necrosis (34-67%) suggests high-grade lesion behavior.",
    },
    "Multifocal or Multicentric": {
        1: "Solitary tumor without additional foci.",
        2: "Multiple enhancing foci are present within the same lobe or hemisphere, consistent with multifocal disease.",
        3: "Tumor foci are distributed across different lobes or hemispheres, indicating multicentric disease.",
        4: "Diffuse infiltration across lobes is consistent with gliomatosis cerebri pattern.",
    },
    "Thickness of enhancing margin": {
        3: "Enhancing rim is thin (<3 mm).",
        4: "Enhancing rim is thick (>3 mm).",
        5: "Lesion demonstrates a solid, uniformly enhancing margin.",
    },
    "Proportion of Oedema": {
        2: "No visible peritumoral edema.",
        3: "Minimal peritumoral edema (<5%) is observed.",
        4: "Mild-to-moderate edema (6-33%) surrounds the lesion.",
        5: "Substantial edema (34-67%) extends into adjacent white matter.",
    },
    "Edema crosses midline": {2: "Edema does not cross the midline.", 3: "Edema extends across the midline."},
    "Ependymal (ventricular) Invasion": {
        1: "No evidence of ependymal invasion or ventricular extension.",
        2: "Tumor breaches the ventricles, suggesting ependymal invasion.",
    },
    "Cortical involvement": {1: "Cortical surface is spared.", 2: "Tumor extends to or involves the cortical surface."},
    "Deep WM invasion": {1: "No evidence of deep white matter invasion.", 2: "Tumor infiltrates deep white matter tracts."},
    "nCET Crosses Midline": {2: "Non-enhancing component does not cross the midline.", 3: "Non-enhancing component extends across the midline."},
    "CET Crosses midline": {2: "Enhancing component does not cross the midline.", 3: "Enhancing component extends across the midline."},
    "Multiple satellites present": {1: "No satellite lesions identified.", 2: "Satellite foci are present adjacent to the main lesion."},
    "Asymmetrical Ventricles": {0: "No tumor-associated ventricular asymmetry.", 1: "Ventricular asymmetry present on the tumor side."},
    "Enlarged Ventricles": {0: "Ventricles are not enlarged.", 1: "Ventricular enlargement is present."},
}

VASARI_CONCISE_MAPS = {
    "Tumor Location": {
        1: "frontal lobe tumor",
        2: "temporal lobe tumor",
        3: "insular tumor",
        4: "parietal lobe tumor",
        5: "occipital lobe tumor",
        6: "brainstem tumor",
        7: "corpus callosum tumor",
        8: "thalamic tumor",
    },
    "Side of Tumor Epicenter": {1: "in the right hemisphere", 2: "in the bilateral involvement", 3: "in the left hemisphere"},
    "Eloquent Brain Involvement": {
        1: "no eloquent cortex involvement",
        2: "speech motor cortex involvement",
        3: "motor cortex involvement",
        4: "visual cortex involvement",
    },
    "Enhancement Quality": {1: "nonenhancing", 2: "mildly enhancing", 3: "mostly enhancing"},
    "Proportion Enhancing": {
        3: "small enhancing proportion (<5%)",
        4: "medium enhancing proportion (6-33%)",
        5: "large enhancing proportion (34-67%)",
        6: "very large enhancing proportion (>68%)",
    },
    "Proportion nCET": {
        3: "small nonenhancing proportion (<5%)",
        4: "mild nonenhancing proportion (6-33%)",
        5: "medium nonenhancing proportion (34-67%)",
        6: "large nonenhancing proportion (68-95%)",
        7: "very large nonenhancing proportion (95-99%)",
        8: "entirely nonenhancing",
    },
    "Proportion Necrosis": {2: "no necrosis", 3: "necrosis <5%", 4: "necrosis 6-33%", 5: "necrosis 34-67%"},
    "Multifocal or Multicentric": {1: "solitary", 2: "multifocal", 3: "multicentric", 4: "gliomatosis pattern"},
    "Thickness of enhancing margin": {
        3: "thin ehancement rim (<3mm)",
        4: "thick enhancement rim (>3mm)",
        5: "solid enhancement with no necrotic core",
    },
    "Proportion of Oedema": {2: "no edema", 3: "small edema (<5%)", 4: "medium edema (6-33%)", 5: "large edema (34-67%)"},
    "Edema crosses midline": {2: "edema confined to one hemisphere", 3: "edema crosses midline"},
    "Ependymal (ventricular) Invasion": {1: "no ependymal invasion", 2: "ependymal invasion present"},
    "Cortical involvement": {1: "no cortical involvement", 2: "cortical involvement present"},
    "Deep WM invasion": {1: "no deep white matter invasion", 2: "deep white matter invasion present"},
    "nCET Crosses Midline": {2: "nonenhancing region does not cross midline", 3: "nonenhancing crosses midline"},
    "CET Crosses midline": {2: "enhancing region does not cross midline", 3: "enhancing region crosses midline"},
    "Multiple satellites present": {1: "no satellite lesions", 2: "satellite lesions present"},
    "Asymmetrical Ventricles": {0: "no tumor-associated ventricular asymmetry", 1: "tumor-side ventricular asymmetry"},
    "Enlarged Ventricles": {0: "ventricles not enlarged", 1: "ventricles enlarged"},
}

VASARI_REASONING_MAPS = {
    "Tumor Location": {
        1: "Most lesion voxels were located within the frontal lobe. Tumor centered in the frontal lobe.",
        2: "Most lesion voxels were located within the temporal lobe. Tumor centered in the temporal lobe.",
        3: "Most lesion voxels were located within the insula. Tumor centered in the insula.",
        4: "Most lesion voxels were located within the parietal lobe. Tumor centered in the parietal lobe.",
        5: "Most lesion voxels were located within the occipital lobe. Tumor centered in the occipital lobe.",
        6: "Most lesion voxels were located within the brainstem. Tumor centered in the brainstem.",
        7: "Most lesion voxels were located within the corpus callosum. Tumor centered in the corpus callosum.",
        8: "Most lesion voxels were located within the thalamus. Tumor centered in the thalamus.",
    },
    "Side of Tumor Epicenter": {
        1: "Lesion volume was greater in the right hemisphere. Tumor predominantly affects the right hemisphere.",
        2: "Lesion voxels were present in both hemispheres. Tumor spans both hemispheres, indicating bilateral involvement.",
        3: "Lesion volume was greater in the left hemisphere. Tumor predominantly affects the left hemisphere.",
    },
    "Eloquent Brain Involvement": {
        1: "No overlap between the lesion mask and eloquent cortex regions. No involvement of eloquent cortical regions.",
        2: "Lesion voxels overlapped with the speech motor region. Tumor involves the speech motor cortex.",
        3: "Lesion voxels overlapped with the primary motor region. Tumor involves motor cortex regions.",
        4: "Lesion voxels overlapped with the visual cortex region. Tumor involves visual cortex regions.",
    },
    "Enhancement Quality": {
        1: "No enhancing voxels were identified in the contrast-enhanced mask. No contrast enhancement is observed.",
        2: "Enhancing voxels were detected but constituted a minor fraction. Mild enhancement is noted.",
        3: "Enhancing voxels formed a major portion of the lesion. Marked enhancement is noted.",
    },
    "Proportion Enhancing": {
        3: "Enhancing voxels constituted less than 5% of total lesion volume. Enhancing component constitutes less than 5% of the lesion.",
        4: "Enhancing voxels comprised 6-33% of total lesion volume. Enhancing component comprises roughly 6-33% of the lesion.",
        5: "Enhancing voxels comprised 34-67% of total lesion volume. Enhancing component involves approximately 34-67% of the lesion volume.",
        6: "Enhancing voxels comprised more than 68% of total lesion volume. Enhancing tissue represents more than 68% of the lesion.",
    },
    "Proportion nCET": {
        3: "Non-enhancing voxels comprised less than 5% of total lesion volume. Non-enhancing tumor component is minimal (<5%).",
        4: "Non-enhancing voxels comprised 6-33% of total lesion volume. Non-enhancing tumor component comprises roughly 6-33% of the lesion.",
        5: "Non-enhancing voxels comprised 34-67% of total lesion volume. Non-enhancing tumor component involves approximately 34-67% of the lesion.",
        6: "Non-enhancing voxels comprised 68-95% of total lesion volume. Non-enhancing tumor component involves 68-95% of the lesion.",
        7: "Non-enhancing voxels comprised 95-99% of total lesion volume. Non-enhancing tumor component occupies 95-99% of the lesion volume.",
        8: "No enhancing voxels were present. Lesion is entirely non-enhancing.",
    },
    "Proportion Necrosis": {
        2: "No necrotic voxels detected within enhancing regions. No visible necrosis within the enhancing lesion.",
        3: "Small necrotic region (<5%) detected within enhancing component. Minimal central necrosis (<5%) is present.",
        4: "Moderate necrotic region (6-33%) detected within enhancing component. Moderate necrosis (6-33%) is identified within the tumor core.",
        5: "Large necrotic region (34-67%) detected within enhancing component. Extensive necrosis (34-67%) suggests high-grade lesion behavior.",
    },
    "Multifocal or Multicentric": {
        1: "Only a single connected enhancing component was detected. Solitary tumor without additional foci.",
        2: "Multiple separate enhancing components were found within the same lobe or hemisphere. Multiple enhancing foci are present within the same lobe or hemisphere, consistent with multifocal disease.",
        3: "Multiple separate enhancing components were detected across different lobes or hemispheres. Tumor foci are distributed across different lobes or hemispheres, indicating multicentric disease.",
        4: "Diffuse enhancement extends across lobes without clear boundaries. Diffuse infiltration across lobes is consistent with gliomatosis cerebri pattern.",
    },
    "Thickness of enhancing margin": {
        3: "Enhancing rim voxel thickness measured less than 3 mm. Enhancing rim is thin (<3 mm).",
        4: "Enhancing rim voxel thickness measured greater than 3 mm. Enhancing rim is thick (>3 mm).",
        5: "Enhancing region was solid and continuous. Lesion demonstrates a solid, uniformly enhancing margin.",
    },
    "Proportion of Oedema": {
        2: "No voxels were labeled as edema. No visible peritumoral edema.",
        3: "Edema voxels constituted less than 5% of total lesion volume. Minimal peritumoral edema (<5%) is observed.",
        4: "Edema voxels comprised 6-33% of total lesion volume. Mild-to-moderate edema (6-33%) surrounds the lesion.",
        5: "Edema voxels comprised more than 34% of total lesion volume. Substantial edema (34-67%) extends into adjacent white matter.",
    },
    "Edema crosses midline": {
        2: "Edema voxels were confined to one hemisphere. Edema does not cross the midline.",
        3: "Edema voxels were detected in both hemispheres. Edema extends across the midline.",
    },
    "Ependymal (ventricular) Invasion": {
        1: "No lesion voxels contacted the ventricular mask. No evidence of ependymal invasion or ventricular extension.",
        2: "Lesion voxels intersected the ventricular mask. Tumor breaches the ventricles, suggesting ependymal invasion.",
    },
    "Cortical involvement": {
        1: "No overlap between lesion voxels and cortical mask. Cortical surface is spared.",
        2: "Lesion voxels overlapped with cortical mask. Tumor extends to or involves the cortical surface.",
    },
    "Deep WM invasion": {
        1: "No lesion voxels overlapped with deep white matter masks. No evidence of deep white matter invasion.",
        2: "Lesion voxels overlapped with deep white matter regions. Tumor infiltrates deep white matter tracts.",
    },
    "nCET Crosses Midline": {
        2: "Non-enhancing component was restricted to one hemisphere. Non-enhancing component does not cross the midline.",
        3: "Non-enhancing component present in both hemispheres. Non-enhancing component extends across the midline.",
    },
    "CET Crosses midline": {
        2: "Enhancing component was restricted to one hemisphere. Enhancing component does not cross the midline.",
        3: "Enhancing component present in both hemispheres. Enhancing component extends across the midline.",
    },
    "Multiple satellites present": {
        1: "Only a single connected enhancing focus was found. No satellite lesions identified.",
        2: "Multiple separate enhancing components were identified adjacent to the main lesion. Satellite foci are present adjacent to the main lesion.",
    },
    "Asymmetrical Ventricles": {
        0: "Ventricle volumes were similar between hemispheres with no evidence of compression. No ventricular asymmetry identified.",
        1: "The ventricle on the tumor side showed reduced volume relative to the contralateral side, consistent with local mass effect. Tumor-associated ventricular asymmetry is present.",
    },
    "Enlarged Ventricles": {
        0: "Measured ventricle volumes were within the expected range without significant dilation. Ventricles are not enlarged.",
        1: "One or both lateral ventricles exceeded the volume threshold indicating dilation. Ventricular enlargement is present.",
    },
}


MAP_STR_TO_MAP = {
    "vasari": VASARI_MAPS,
    "vasari_sentence": VASARI_SENTENCE_MAPS,
    "vasari_reasoning": VASARI_REASONING_MAPS,
    "vasari_concise": VASARI_CONCISE_MAPS,
}


