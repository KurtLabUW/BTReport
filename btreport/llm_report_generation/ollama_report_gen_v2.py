import json
import os
import ollama
from pathlib import Path
from tqdm import tqdm
import argparse
import unicodedata
from sanitext.text_sanitization import sanitize_text, get_allowed_characters

EXAMPLE_FINDINGS = """
1.
FINDINGS:
MASS EFFECT & VENTRICLES: There is at least 8 mm the left-to-right midline shift at
the level of the foramen of Monro. There is medialization of the left uncus without 
frank herniation. The right lateral ventricle is enlarged, concerning for entrapment. 
There is effacement of the frontal horn of the left lateral ventricle. There is left 
frontal sulcal effacement. No tonsillar herniation. The basal cisterns are patent.
BRAIN/ENHANCEMENT: Within the left frontal lobe there is a rim-enhancing mass 
which measures 3.4 x 3.2 x 3.5 cm. There is no associated restricted diffusion. There is 
a focus of T2 hyperintensity centrally within the mass which may represent necrosis. 
There are foci of susceptibility which may represent intratumoral bleeding. There is a 
large volume of surrounding vasogenic edema.

2.
FINDINGS:
MASS EFFECT & VENTRICLES: The below described right parieto-occipital enhancing lesion exerts significant mass effect on the posterior atrium of the right lateral ventricle. Approximately 1.1 cm of right-to-left midline shift with broad diffuse effacement of the right-sided cerebral sulci. The cisterns are normal.
BRAIN:  Right parieto-occipital lobe intrinsically T1 bright mass measuring 5.4 x 4.1 x 3.5 cm (1103/57, 501/116), with peripheral rim enhancement which is nodular-like at the posterior medial margins. Diffusion restriction, with signal loss on ADC throughout the peripheral borders of the mass. Multiple punctate foci of SWI signal abnormalities compatible with hemosiderin deposition. Surrounding FLAIR abnormality likely from vasogenic edema within the right frontoparietal lobes with inferior extension into the superior right temporal lobe. Additionally the FLAIR signal abnormality crosses the posterior portion of the corpus callosum, including the body and splenium, and abuts the left posterior lateral ventricle.

3. 
 FINDINGS:
MASS EFFECT & VENTRICLES: Effacement of the anterior horns of the lateral ventricles. There is approximately 5 mm of rightward midline shift of the anterior brain.
BRAIN:  There is a large enhancing lesion, be originating in the anterior paramedian left frontal lobe and crossing the corpus callosum. The lesion invades into the anterior aspect of the lateral ventricles and has multiple small enhancing satellite lesions in both frontal lobes. There is a large necrotic area, which facilitates diffusion, in the left frontal component which measures up to 2.5 cm.


"""

REPORT_TEMPLATE = """
You are a radiologist generating a synthetic clinical MRI report.

Below are example FINDINGS sections taken from real brain tumor reports:

EXAMPLE FINDINGS:
{example_findings}

---

Your job is to generate a FINDINGS section in the same clinical style, but only using the metadata provided below.

Please abide strictly to the following rules (follow them exactly).

In your findings section:

1) Use only the metadata provided for quantitative statements. Do NOT hallucinate any information that is not directly inferable.  
2) Include 10-20 clinically meaningful findings summarized in an anatomically descriptive manner. Prioritize describing abnormal or clinically significant observations.
3) Preserve the subsection structure from the example reports. Make sure to include the following subsections:
    a) MASS EFFECT & VENTRICLES
    b) BRAIN / ENHANCEMENT
4) Never mention imaging sequences other than T1n, T2w, T2 FLAIR, or T1-Gd. Do not mention diffusion, perfusion, spectroscopy, MRA, or other modalities (unless stated explicitly in metadata). 
5) Do not mention structures or measurements not present in the metadata.
6) Mandatory Considerations: Make sure to include the following findings below if present in metadata. Remember to follow the sentence structure in the example reports.  
    a) Maximum midline shift represented in mm units.
        i) Make sure to describe the magnitude and the direction of the midline shift.
        ii) Describe the anatomical level at which the (e.g., foramen of Monro, third and fourth ventricles, septum pellucidum).
        iii) If shift is minimal (e.g., < 5mm), explicitly state the measurement as no shift, but still provide the measurement.
    b) If tumor mass effect is present, describe the mass effect on ventricles or surrounding brain structures.
        i) Include a description of ventricular effacement (if present), including which horn (anterior/posterior horn), and on which hemisphere it is observed.
    c) Comment on ventricular status. If effacement is present, describe the extent of the effacement or asymmetry. If ventricles are normal, explicitly state so (mirroring example reports). Use the following metadata fields in your description: "Asymmetrical Ventricles", "Enlarged Ventricles"
    d) Describe the size of the primary lesion, as well as any smaller secondary lesions (if present) represented in cm units. Use the 3D measurements from the metadata. Make sure to include the following:
        i) If multiple lesions exist, summarize number, dominant lesion, and laterality. Use the following metadata fields in your description: "Number of lesions" and "Multifocal or multicentric."
        ii) Anatomical location of lesion(s)
        iii) Use the following metadata fields in your description: "Tumor Location", "Side of Tumor Epicenter", and "Region Proportions."
    e) Describe enhancing characteristics. Use the following metadata fields in your description: "Enhancement quality", "Thickness of enhancing margin", "Proportion Enhancing"
        i) Describe enhancement style (e.g., rim-enhancing, mildly enhancing, peripheral enhancing, multi-lobular enhancing) only if explicitly supported.
        ii) Describe edematous tissue. Use the following metadata fields in your description: "ED volume", Whether edema crosses midline, "Proportion of edema"
        iii) Describe vasogenic edema and its extent only if metadata supports it.
    f) Describe invasion and involvement. Use the following metadata fields in your description: “Cortical involvement”, “Deep WM invasion”, “Ependymal invasion”, “Eloquent Brain Involvement”
    g) Describe the necrosis if present. Use the metadata field "Proportion Necrosis" to describe the central foci of necrosis.

METADATA (for subject {subject_id}):
{metadata_json}

---

Write the FINDINGS section now, using clinical radiology language.
"""



REPORT_TEMPLATE_IMAGE = """
You are a radiologist generating a synthetic clinical MRI report.

Below are example FINDINGS sections taken from real brain tumor reports:

EXAMPLE FINDINGS:
{example_findings}

---

Your job is to generate a FINDINGS section in the same clinical style, but only using the metadata provided below.

Please abide strictly to the following rules (follow them exactly).

In your findings section:

1) Use only the metadata provided for quantitative statements. Do NOT hallucinate any information that is not directly inferable.  
2) Include 10-20 clinically meaningful findings summarized in an anatomically descriptive manner. Prioritize describing abnormal or clinically significant observations.
3) Preserve the subsection structure from the example reports. Make sure to include the following subsections:
    a) MASS EFFECT & VENTRICLES
    b) BRAIN / ENHANCEMENT
4) Never mention imaging sequences other than T1n, T2w, T2 FLAIR, or T1-Gd. Do not mention diffusion, perfusion, spectroscopy, MRA, or other modalities (unless stated explicitly in metadata). 
5) Do not mention structures or measurements not present in the metadata, or in the image.
6) Mandatory Considerations: Make sure to include the following findings below if present in metadata. Remember to follow the sentence structure in the example reports.  
    a) Maximum midline shift represented in mm units.
        i) Make sure to describe the magnitude and the direction of the midline shift.
        ii) Describe the anatomical level at which the (e.g., foramen of Monro, third and fourth ventricles, septum pellucidum).
        iii) If shift is minimal (e.g., < 5mm), explicitly state the measurement as no shift, but still provide the measurement.
    b) If tumor mass effect is present, describe the mass effect on ventricles or surrounding brain structures.
        i) Include a description of ventricular effacement (if present), including which horn (anterior/posterior horn), and on which hemisphere it is observed.
    c) Comment on ventricular status. If effacement is present, describe the extent of the effacement or asymmetry. If ventricles are normal, explicitly state so (mirroring example reports). Use the following metadata fields in your description: "Asymmetrical Ventricles", "Enlarged Ventricles"
    d) Describe the size of the primary lesion, as well as any smaller secondary lesions (if present) represented in cm units. Use the 3D measurements from the metadata. Make sure to include the following:
        i) If multiple lesions exist, summarize number, dominant lesion, and laterality. Use the following metadata fields in your description: "Number of lesions" and "Multifocal or multicentric."
        ii) Anatomical location of lesion(s)
        iii) Use the following metadata fields in your description: "Tumor Location", "Side of Tumor Epicenter", and "Region Proportions."
    e) Describe enhancing characteristics. Use the following metadata fields in your description: "Enhancement quality", "Thickness of enhancing margin", "Proportion Enhancing"
        i) Describe enhancement style (e.g., rim-enhancing, mildly enhancing, peripheral enhancing, multi-lobular enhancing) only if explicitly supported.
        ii) Describe edematous tissue. Use the following metadata fields in your description: "ED volume", Whether edema crosses midline, "Proportion of edema"
        iii) Describe vasogenic edema and its extent only if metadata supports it.
    f) Describe invasion and involvement. Use the following metadata fields in your description: “Cortical involvement”, “Deep WM invasion”, “Ependymal invasion”, “Eloquent Brain Involvement”
    g) Describe the necrosis if present. Use the metadata field "Proportion Necrosis" to describe the central foci of necrosis.

IMAGE:
{image_path}
METADATA (for subject {subject_id}):
{metadata_json}

---

Write the FINDINGS section now, using clinical radiology language.
"""




def generate_llm_report(subject_id, metadata, image_path=None, model="gpt-oss:120b"):

    if image_path is None:
        prompt = REPORT_TEMPLATE.format(
            example_findings=EXAMPLE_FINDINGS,
            subject_id=subject_id,
            metadata_json=json.dumps(metadata, indent=2),
        )
    else:
        prompt = REPORT_TEMPLATE_IMAGE.format(
            example_findings=EXAMPLE_FINDINGS,
            image_path=image_path,
            subject_id=subject_id,
            metadata_json=json.dumps(metadata, indent=2),
        )     

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    report=response["message"]["content"]
    report = report.replace('\u2011', '-')
    return sanitize_text(report) 

