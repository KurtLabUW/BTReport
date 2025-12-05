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

Now generate a similar FINDINGS section, but ONLY using the metadata provided below.

Do NOT hallucinate any information that is not directly inferable.  
Preserve all of the subsections in the example findings reports
Choose the top 10 metadata entries, as true reports report around 7-10 facts.
Prioritize abnormal or clinically significant findings.
Only T1n, T2w, T2 Flair, and T1-Gd were obtained, so do not comment on features like diffusion.
Remember to comment on midline shift in the style of the reports. Make sure to comment if there is a mass effect.
Comment on the effacement of ventricles if present according to the metadata, and give details on the side and if the effacement is in the anterior/posterior horns.
Comment on the dimension of the lesion(s) in 3D in cm.
Do NOT mention measurements, structures, or features unless supported by the metadata.  
Do NOT mention modalities or sequences not given.  
The goal is to produce a realistic, readable FINDINGS section grounded purely in the structured fields.

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

Now generate a similar FINDINGS section, but ONLY using the metadata provided below and the T1c MRI image.

Do NOT hallucinate any information that is not directly inferable.  
Preserve all of the subsections in the example findings reports
Choose the top 10 metadata entries, as true reports report around 7-10 facts.
Prioritize abnormal or clinically significant findings.
Only T1n, T2w, T2 Flair, and T1-Gd were obtained, so do not comment on features like diffusion.
Remember to comment on midline shift in the style of the reports.
Comment on the effacement of ventricles if present according to the metadata, and give details on the side and if the effacement is in the anterior/posterior horns.
Do NOT mention measurements, structures, or features unless supported by the metadata.  
Do NOT mention modalities or sequences not given.  
The goal is to produce a realistic, readable FINDINGS section grounded purely in the structured fields.

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

