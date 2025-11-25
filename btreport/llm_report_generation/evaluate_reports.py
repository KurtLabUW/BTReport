from RadEval import RadEval
import json


evaluator = RadEval(
    do_rouge=True,
    do_bertscore=True,
    do_srr_bert=True,
    do_ratescore=True,
    # do_details=True
)

#  Single reference FINDINGS
reference_text = """FINDINGS:
MASS EFFECT & VENTRICLES: There is at least 8 mm of left-to-right midline shift at the level of the foramen of Monro. 
There is medialization of the left uncus without frank herniation. The right lateral ventricle is enlarged, concerning for entrapment. 
There is effacement of the frontal horn of the left lateral ventricle and left frontal sulcal effacement. 
No tonsillar herniation. The basal cisterns are patent.
BRAIN/ENHANCEMENT: A rim-enhancing mass is present in the left frontal lobe measuring 3.4 × 3.2 × 3.5 cm. 
There is no restricted diffusion. Central T2 hyperintensity may represent necrosis. 
Susceptibility foci may reflect intratumoral bleeding. Large vasogenic edema is present.
VASCULAR: Intracranial vascular flow voids are normal.
EXTRA-AXIAL: Extra-axial spaces are normal.
EXTRA-CRANIAL: Skull and facial bones are normal. Fluid is present within the left maxillary sinus. Orbits are normal.
"""

# Repeat the same reference for each hypothesis
refs = [reference_text, reference_text, reference_text]

#  Candidate prompts being evaluated
hyps = [
    """FINDINGS:
There is a mild left-to-right midline shift measuring up to approximately 5 mm at the septum pellucidum. 
The ventricles are symmetric without enlargement or entrapment. 
Within the left temporal lobe, there is a solitary tumor demonstrating mild enhancement with an enhancing margin greater than 3 mm. 
A small proportion of the lesion is necrotic, and it is surrounded by a large volume of vasogenic edema. 
The lesion does not cross the midline and shows no cortical, deep white matter, or ependymal invasion.""",
    """FINDINGS:
A left temporal lobe lesion is present with a substantial volume of vasogenic edema comprising a large portion of the total tumor burden. 
There is mild mass effect with midline shift between 2 and 5 mm and no ventricular asymmetry or entrapment. 
Enhancement is mild and involves only a small fraction of the tumor, with minimal necrosis and no satellite lesions. 
The lesion remains confined to the left temporal lobe without midline crossing or invasion of deep white matter, cortex, or ependymal surfaces.""",
    """FINDINGS:
A solitary left temporal tumor demonstrates mild enhancement and a small degree of necrosis. 
A large surrounding region of vasogenic edema contributes to a mild midline shift of up to approximately 5 mm. 
The ventricles are not enlarged or asymmetric, and no ventricular entrapment is suggested. 
No midline crossing, satellite lesions, deep white matter invasion, cortical involvement, or ependymal extension are indicated by the available features.""",
]

results = evaluator(refs=refs, hyps=hyps)
print(json.dumps(results, indent=2))
