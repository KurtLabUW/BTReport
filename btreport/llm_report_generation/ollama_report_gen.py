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
MASS EFFECT & VENTRICLES: The below described right parieto-occipital enhancing lesion exerts significant mass effect on the posterior atrium of the right lateral ventricle. Approximately 1.1 cm of right-to-left midline shift with broad diffuse effacement of the right-sided cerebral sulci. The cisterns are normal.
BRAIN:  Right parieto-occipital lobe intrinsically T1 bright mass measuring 5.4 x 4.1 x 3.5 cm (1103/57, 501/116), with peripheral rim enhancement which is nodular-like at the posterior medial margins. Diffusion restriction, with signal loss on ADC throughout the peripheral borders of the mass. Multiple punctate foci of SWI signal abnormalities compatible with hemosiderin deposition. Surrounding FLAIR abnormality likely from vasogenic edema within the right frontoparietal lobes with inferior extension into the superior right temporal lobe. Additionally the FLAIR signal abnormality crosses the posterior portion of the corpus callosum, including the body and splenium, and abuts the left posterior lateral ventricle.



"""

REPORT_TEMPLATE = """
You are a radiologist generating a synthetic clinical MRI report.

Below are example FINDINGS sections taken from real brain tumor reports:

EXAMPLE FINDINGS:
{example_findings}

---

Now generate a similar FINDINGS section, but ONLY using the metadata provided below.

Do NOT hallucinate any information that is not directly inferable.  
Only T1n, T2w, T2 Flair, and T1-Gd were obtained, so do not comment on features like diffusion.
Do NOT mention measurements, structures, or features unless supported by the metadata.  
Do NOT mention modalities or sequences not given.  
The goal is to produce a realistic, readable FINDINGS section grounded purely in the structured fields.

METADATA (for subject {subject_id}):
{metadata_json}

---

Write the FINDINGS section now, using clinical radiology language.
"""


def generate_llm_report(subject_id, metadata, model="gpt-oss:120b"):
    prompt = REPORT_TEMPLATE.format(
        example_findings=EXAMPLE_FINDINGS,
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


# TODO: add merge logic
def main(args):
    model_name = args.model_name
    num_splits = args.num_splits
    save_every = args.save_every
    split_no = args.split_no

    input_json = "brats23_metadata.json"
    outdir = f"reports/{model_name}"
    os.makedirs(outdir, exist_ok=True)
    output_json = f"{outdir}/brats23_metadata-report-{model_name.replace(':','-')}-split{split_no}of{num_splits}.json"

    # load all subjects
    with open(input_json, "r") as f:
        subjects = list(json.load(f).items())

    # slice all subjects for this split
    total = len(subjects)
    per_split = total // num_splits + int(total % num_splits > 0)
    start = (split_no - 1) * per_split
    end = min(start + per_split, total)
    subjects = subjects[start:end]

    # resume from existing
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            output = json.load(f)
    else:
        output = {}

    processed = 0

    # iterate through subjects in this split
    for idx, (subject_id, metadata) in enumerate(tqdm(subjects, desc="Subjects"), start=1):
        if subject_id in output:  # skip subjects already done
            continue

        print(f"Processing {idx} / {len(subjects)}: {subject_id}")
        findings = generate_report(subject_id, metadata, model_name)
        output[subject_id] = {"findings": findings, "metadata": metadata}
        print(f"Finished {subject_id}")

        processed += 1

        # periodic save
        if processed % save_every == 0:
            tmp = output_json + ".tmp"
            with open(tmp, "w") as f:
                json.dump(output, f, indent=2)
            os.replace(tmp, output_json)
            print(f"Saved after {processed} new subjects")

    # final save
    tmp = output_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp, output_json)

    print("Final save complete.")


# model_name="gpt-oss:120b"
# # model_name="deepseek-r1:32b"

# python ollama_report_gen.py --num_splits 4 --split_no 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="gpt-oss:120b")
    parser.add_argument("--save_every", type=int, default=15)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--split_no", type=int, default=1)

    args = parser.parse_args()

    main(args)
