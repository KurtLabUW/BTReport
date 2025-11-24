import pandas as pd
import re
from pathlib import Path
from functools import reduce
import json
import numpy as np
import argparse


def flatten_json(nested_json, exclude=['']):
    """Flatten a nested JSON object into a single level dict, preserving lists. 
    Inspired by https://stackoverflow.com/questions/52795561/flattening-nested-json-in-pandas-data-frame"""
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                if a not in exclude:
                    flatten(x[a], f"{name}{a}__")
        elif isinstance(x, list):
            # Preserve the list as-is
            out[name[:-2] if name.endswith('__') else name[:-1]] = x
        else:
            out[name[:-2] if name.endswith('__') else name[:-1]] = x

    flatten(nested_json)
    return out



def merge_datasheets(merged_save_path='./datasheets/all_merged.csv'):
    # GBM mapping
    ref_csv="./datasheets/GBM_Subjects_Spreadsheet.xlsx"
    ref_df = pd.read_excel(ref_csv, sheet_name="CLINICAL-RAW-DATA-FINAL")
    mapping_gbm = {}
    for _, row in ref_df.iterrows():
        brats23 = row["2023_subject_id"]
        brats21 = row["2021_subject_id"]
        if pd.notna(brats23) and pd.notna(brats21):
            mapping_gbm[brats23] = brats21
            
    # TCGA-TCIA mapping
    ref_csv="./datasheets/TCGA-TCIA-BraTS-Mapping.xlsx"
    ref_df = pd.read_excel(ref_csv, sheet_name="Training")
    mapping_tcga_tcia = {}
    for _, row in ref_df.iterrows():
        tcia = row["TCIA_ID"]
        brats21 = row["BraTS_2021_ID"]
        if pd.notna(tcia) and pd.notna(brats21):
            mapping_tcga_tcia[tcia] = brats21
            


    # CPTAC mapping
    ref_csv="./datasheets/GBM_Subjects_Spreadsheet.xlsx"
    ref_df = pd.read_excel(ref_csv, sheet_name="CLINICAL-RAW-DATA-FINAL")
    mapping_cptac = {}
    for _, row in ref_df.iterrows():
        id = row["dataset_id"]
        brats21 = row["2021_subject_id"]
        if pd.notna(brats23) and pd.notna(brats21) and "CPTAC" in row["dataset"]:
            mapping_cptac[id] = brats21

    # UPENN mapping
    ref_csv="./datasheets/GBM_Subjects_Spreadsheet.xlsx"
    ref_df = pd.read_excel(ref_csv, sheet_name="CLINICAL-RAW-DATA-FINAL")
    mapping_upenn = {}
    for _, row in ref_df.iterrows():
        id = row["dataset_id"]
        brats21 = row["2021_subject_id"]
        if pd.notna(brats23) and pd.notna(brats21) and "upenn" in row["dataset"].lower():
            mapping_upenn[id] = brats21


    mappings = {
        "gbm": mapping_gbm,
        "tcga_tcia": mapping_tcga_tcia,
        "cptac": mapping_cptac,
        "upenn": mapping_upenn,
        
    }

    def normalize_id_midline(x, mapping_key='gbm'):
        if pd.isna(x): return None
        sp = x.split('-')[:4]
        s = '-'.join(sp)
        s = mappings[mapping_key].get(s, None) 

        return s


    def normalize_id_tcga_tcia(x, mapping_key='tcga_tcia'):
        if pd.isna(x): return None
        s = mappings[mapping_key].get(x, None) 
        return s

    def normalize_id_upenn(x, mapping_key='upenn'):
        if pd.isna(x): return None
        s = mappings[mapping_key].get(x, None) 
        return s

    def normalize_id_cptac(x, mapping_key='cptac'):
        if pd.isna(x): return None
        s = mappings[mapping_key].get(x, None) 
        return s


    def normalize_id_identity(x):
        if pd.isna(x): return None
        return x

    normalizers = {
        "subject": normalize_id_midline,
        "2021_subject_id": normalize_id_identity,
        "BraTS21 ID": normalize_id_identity,
        "case_submitter_id": normalize_id_tcga_tcia,
        "BraTS_2021_ID": normalize_id_identity,
        "ID": normalize_id_upenn,
        "Cases Submitter ID": normalize_id_cptac,
        "id": normalize_id_midline,
    }

    def load_csv(path, id_col, sheet_name=None ):
        if sheet_name is not None:
            df=pd.read_excel(path, sheet_name=sheet_name)
        else:
            df = pd.read_csv(path)
        df = df.rename(columns={id_col: "id"})
        df["id"] = df["id"].map(normalizers.get(id_col, lambda x: x))
        prefix = Path(path).stem
        new_cols = {c: f"{prefix}__{c}" for c in df.columns if c != "id"}
        return df.rename(columns=new_cols).drop_duplicates(subset=["id"])

    def merge_csvs(csv_to_idcol):
        dfs = [load_csv(p, id_col, sheet) for p, id_col, sheet in csv_to_idcol]
        return reduce(lambda l, r: pd.merge(l, r, on="id", how="outer"), dfs)


    csv_to_idcol = [
        ("./datasheets/midline_summary.csv", "subject", None),
        ("./datasheets/GBM_Subjects_Spreadsheet.xlsx", "2021_subject_id", "CLINICAL-RAW-DATA-FINAL"),
        ("./datasheets/GBM_Subjects_Spreadsheet.xlsx", "2021_subject_id", "IMAGING-DATA"),
        ("./datasheets/brats23_metadata_flattened.csv", "id", None),
        ("./datasheets/UCSF-PDGM-Clinical.csv", "BraTS21 ID", None),
        ("./datasheets/UCSF-PDGM-metadata_v5.csv", "BraTS21 ID", None),
        ("./datasheets/tcga-tcia.xlsx", "case_submitter_id", "clinical"),
        ("./datasheets/BraTS Missing IDs.xlsx", "BraTS_2021_ID", "Sheet1"),
        ("./datasheets/UPENN-GBM_Clinical.csv", "ID", None),
        ("./datasheets/CPTAC-GBM-Clinical.csv", "Cases Submitter ID", None),
    ]
    merged = merge_csvs(csv_to_idcol)
    merged.to_csv(merged_save_path, index=False)
    print(f'Datasheets merged!:')
    print(f'Merged df contains {merged.shape[0]} rows and {merged.shape[1]} columns.')
    print('First 10 keys', list(merged.keys())[:10], '\n')
    return merged

cols_to_delete = [
    'GBM_Subjects_Spreadsheet__resection_site',
    'GBM_Subjects_Spreadsheet__age_at_diagnosis',	
    'GBM_Subjects_Spreadsheet__days_to_birth',	
    'GBM_Subjects_Spreadsheet__birth_year',	
    'GBM_Subjects_Spreadsheet__death_year',	
    'GBM_Subjects_Spreadsheet__last_fu_days',	
    'GBM_Subjects_Spreadsheet__last_disease_days',	
    'GBM_Subjects_Spreadsheet__days_to_recurrence',
    'GBM_Subjects_Spreadsheet__progression_y_n',
    "tcga-tcia__age_is_obfuscated", "tcga-tcia__cause_of_death", "tcga-tcia__cause_of_death_source",
    "tcga-tcia__country_of_residence_at_enrollment", "tcga-tcia__occupation_duration_years", "tcga-tcia__premature_at_birth",
    "tcga-tcia__weeks_gestation_at_birth", "tcga-tcia__adrenal_hormone", "tcga-tcia__ajcc_clinical_m",
    "tcga-tcia__ajcc_clinical_n", "tcga-tcia__ajcc_clinical_stage", "tcga-tcia__ajcc_clinical_t",
    "tcga-tcia__ajcc_pathologic_m", "tcga-tcia__ajcc_pathologic_n", "tcga-tcia__ajcc_pathologic_stage",
    "tcga-tcia__ajcc_pathologic_t", "tcga-tcia__ajcc_staging_system_edition", "tcga-tcia__anaplasia_present",
    "tcga-tcia__anaplasia_present_type", "tcga-tcia__ann_arbor_b_symptoms", "tcga-tcia__ann_arbor_b_symptoms_described",
    "tcga-tcia__ann_arbor_clinical_stage", "tcga-tcia__ann_arbor_extranodal_involvement", "tcga-tcia__ann_arbor_pathologic_stage",
    "tcga-tcia__best_overall_response", "tcga-tcia__breslow_thickness", "tcga-tcia__burkitt_lymphoma_clinical_variant",
    "tcga-tcia__child_pugh_classification", "tcga-tcia__circumferential_resection_margin", "tcga-tcia__cog_liver_stage",
    "tcga-tcia__cog_neuroblastoma_risk_group", "tcga-tcia__cog_renal_stage", "tcga-tcia__cog_rhabdomyosarcoma_risk_group",
    "tcga-tcia__days_to_best_overall_response", "tcga-tcia__days_to_last_known_disease_status", "tcga-tcia__days_to_recurrence",
    "tcga-tcia__eln_risk_classification", "tcga-tcia__enneking_msts_grade", "tcga-tcia__enneking_msts_metastasis",
    "tcga-tcia__enneking_msts_stage", "tcga-tcia__enneking_msts_tumor_site", "tcga-tcia__esophageal_columnar_dysplasia_degree",
    "tcga-tcia__esophageal_columnar_metaplasia_present", "tcga-tcia__figo_stage", "tcga-tcia__figo_staging_edition_year",
    "tcga-tcia__first_symptom_prior_to_diagnosis", "tcga-tcia__gastric_esophageal_junction_involvement", "tcga-tcia__gleason_grade_group",
    "tcga-tcia__gleason_grade_tertiary", "tcga-tcia__gleason_patterns_percent", "tcga-tcia__goblet_cells_columnar_mucosa_present",
    "tcga-tcia__greatest_tumor_dimension", "tcga-tcia__gross_tumor_weight", "tcga-tcia__igcccg_stage",
    "tcga-tcia__inpc_grade", "tcga-tcia__inpc_histologic_group", "tcga-tcia__inrg_stage",
    "tcga-tcia__inss_stage", "tcga-tcia__international_prognostic_index", "tcga-tcia__irs_group",
    "tcga-tcia__irs_stage", "tcga-tcia__ishak_fibrosis_score", "tcga-tcia__iss_stage",
    "tcga-tcia__largest_extrapelvic_peritoneal_focus", "tcga-tcia__laterality", "tcga-tcia__lymph_node_involved_site",
    "tcga-tcia__lymph_nodes_positive", "tcga-tcia__lymph_nodes_tested", "tcga-tcia__lymphatic_invasion_present",
    "tcga-tcia__margin_distance", "tcga-tcia__margins_involved_site", "tcga-tcia__masaoka_stage",
    "tcga-tcia__medulloblastoma_molecular_classification", "tcga-tcia__metastasis_at_diagnosis", "tcga-tcia__metastasis_at_diagnosis_site",
    "tcga-tcia__method_of_diagnosis", "tcga-tcia__micropapillary_features", "tcga-tcia__mitosis_karyorrhexis_index",
    "tcga-tcia__mitotic_count", "tcga-tcia__non_nodal_regional_disease", "tcga-tcia__non_nodal_tumor_deposits",
    "tcga-tcia__ovarian_specimen_status", "tcga-tcia__ovarian_surface_involvement", "tcga-tcia__papillary_renal_cell_type",
    "tcga-tcia__percent_tumor_invasion", "tcga-tcia__perineural_invasion_present", "tcga-tcia__peripancreatic_lymph_nodes_positive",
    "tcga-tcia__peripancreatic_lymph_nodes_tested", "tcga-tcia__peritoneal_fluid_cytological_status", "tcga-tcia__pregnant_at_diagnosis",
    "tcga-tcia__primary_disease", "tcga-tcia__primary_gleason_grade", "tcga-tcia__residual_disease",
    "tcga-tcia__satellite_nodule_present", "tcga-tcia__secondary_gleason_grade", "tcga-tcia__sites_of_involvement",
    "tcga-tcia__supratentorial_localization", "tcga-tcia__transglottic_extension", "tcga-tcia__tumor_confined_to_organ_of_origin",
    "tcga-tcia__tumor_depth", "tcga-tcia__tumor_focality", "tcga-tcia__tumor_largest_dimension_diameter",
    "tcga-tcia__tumor_regression_grade", "tcga-tcia__tumor_stage", "tcga-tcia__vascular_invasion_present",
    "tcga-tcia__vascular_invasion_type", "tcga-tcia__weiss_assessment_score", "tcga-tcia__who_cns_grade",
    "tcga-tcia__who_nte_grade", "tcga-tcia__wilms_tumor_histologic_subtype", "tcga-tcia__chemo_concurrent_to_radiation",
    "tcga-tcia__days_to_treatment_end", "tcga-tcia__days_to_treatment_start", "tcga-tcia__initial_disease_status",
    "tcga-tcia__number_of_cycles", "tcga-tcia__reason_treatment_ended", "tcga-tcia__regimen_or_line_of_therapy",
    "tcga-tcia__route_of_administration", "tcga-tcia__therapeutic_agents", "tcga-tcia__treatment_anatomic_site",
    "tcga-tcia__treatment_arm", "tcga-tcia__treatment_dose", "tcga-tcia__treatment_dose_units",
    "tcga-tcia__treatment_effect", "tcga-tcia__treatment_effect_indicator", "tcga-tcia__treatment_frequency",
    "tcga-tcia__treatment_intent_type", "tcga-tcia__treatment_outcome", "UPENN-GBM_Clinical__PsP_TP_score",
    "CPTAC-GBM-Clinical__Related Entities", "CPTAC-GBM-Clinical__Annotation", "CPTAC-GBM-Clinical__Prior Malignancy",
    "CPTAC-GBM-Clinical__AJCC Clinical M", "CPTAC-GBM-Clinical__AJCC Clinical N", "CPTAC-GBM-Clinical__AJCC Clinical Stage",
    "CPTAC-GBM-Clinical__AJCC Clinical T", "CPTAC-GBM-Clinical__AJCC Pathologic M", "CPTAC-GBM-Clinical__AJCC Pathologic N",
    "CPTAC-GBM-Clinical__AJCC Pathologic T", "CPTAC-GBM-Clinical__Ann Arbor B Symptoms", "CPTAC-GBM-Clinical__Ann Arbor Clinical Stage",
    "CPTAC-GBM-Clinical__Ann Arbor Extranodal Involvement", "CPTAC-GBM-Clinical__Ann Arbor Pathologic Stage", "CPTAC-GBM-Clinical__Best Overall Response",
    "CPTAC-GBM-Clinical__Burkitt Lymphoma Clinical Variant", "CPTAC-GBM-Clinical__Circumferential Resection Margin", "CPTAC-GBM-Clinical__Colon Polups History",
    "CPTAC-GBM-Clinical__Days to Best Overall", "CPTAC-GBM-Clinical__Days to Diagnosis", "CPTAC-GBM-Clinical__Days to HIV Diagnosis",
    "CPTAC-GBM-Clinical__Days to New Event", "CPTAC-GBM-Clinical__Figo Stage", "CPTAC-GBM-Clinical__HIV Positive",
    "CPTAC-GBM-Clinical__HPV Positive Type", "CPTAC-GBM-Clinical__HPV Status", "CPTAC-GBM-Clinical__ISS Stage",
    "CPTAC-GBM-Clinical__Laterality", "CPTAC-GBM-Clinical__LDH Level at Diagnosis", "CPTAC-GBM-Clinical__LDH Normal Range Upper",
    "CPTAC-GBM-Clinical__Lymph Nodes Positive", "CPTAC-GBM-Clinical__Lymphatic Invasion Present", "CPTAC-GBM-Clinical__Method of Diagnosis",
    "CPTAC-GBM-Clinical__New Event Anatomic Site", "CPTAC-GBM-Clinical__New Event Type", "CPTAC-GBM-Clinical__Overall Survival",
    "CPTAC-GBM-Clinical__Perineural Invasion Present", "CPTAC-GBM-Clinical__Prior Treatment", "CPTAC-GBM-Clinical__Progression Free Survival",
    "CPTAC-GBM-Clinical__Progression Free Survival Event", "CPTAC-GBM-Clinical__Residual Disease", "CPTAC-GBM-Clinical__Vascular Invasion Present",
    "CPTAC-GBM-Clinical__Age At Index", "CPTAC-GBM-Clinical__Premature At Birth", "CPTAC-GBM-Clinical__Weeks Gestation At Birth",
    "CPTAC-GBM-Clinical__Age Is Obfuscated", "CPTAC-GBM-Clinical__Cause Of Death Source", "CPTAC-GBM-Clinical__Occupation Duration Years",
    "CPTAC-GBM-Clinical__Country Of Residence At Enrollment", "CPTAC-GBM-Clinical__Icd 10 Code", "CPTAC-GBM-Clinical__Synchronous Malignancy",
    "CPTAC-GBM-Clinical__Anaplasia Present", "CPTAC-GBM-Clinical__Anaplasia Present Type", "CPTAC-GBM-Clinical__Child Pugh Classification",
    "CPTAC-GBM-Clinical__Cog Liver Stage", "CPTAC-GBM-Clinical__Cog Neuroblastoma Risk Group", "CPTAC-GBM-Clinical__Cog Renal Stage",
    "CPTAC-GBM-Clinical__Cog Rhabdomyosarcoma Risk Group", "CPTAC-GBM-Clinical__Enneking Msts Grade", "CPTAC-GBM-Clinical__Enneking Msts Metastasis",
    "CPTAC-GBM-Clinical__Enneking Msts Stage", "CPTAC-GBM-Clinical__Enneking Msts Tumor Site", "CPTAC-GBM-Clinical__Esophageal Columnar Dysplasia Degree",
    "CPTAC-GBM-Clinical__Esophageal Columnar Metaplasia Present", "CPTAC-GBM-Clinical__First Symptom Prior To Diagnosis", "CPTAC-GBM-Clinical__Gastric Esophageal Junction Involvement",
    "CPTAC-GBM-Clinical__Goblet Cells Columnar Mucosa Present", "CPTAC-GBM-Clinical__Gross Tumor Weight", "CPTAC-GBM-Clinical__Inpc Grade",
    "CPTAC-GBM-Clinical__Inpc Histologic Group", "CPTAC-GBM-Clinical__Inrg Stage", "CPTAC-GBM-Clinical__Inss Stage",
    "CPTAC-GBM-Clinical__Irs Group", "CPTAC-GBM-Clinical__Irs Stage", "CPTAC-GBM-Clinical__Ishak Fibrosis Score",
    "CPTAC-GBM-Clinical__Lymph Nodes Tested", "CPTAC-GBM-Clinical__Medulloblastoma Molecular Classification", "CPTAC-GBM-Clinical__Metastasis At Diagnosis",
    "CPTAC-GBM-Clinical__Metastasis At Diagnosis Site", "CPTAC-GBM-Clinical__Mitosis Karyorrhexis Index", "CPTAC-GBM-Clinical__Peripancreatic Lymph Nodes Positive",
    "CPTAC-GBM-Clinical__Peripancreatic Lymph Nodes Tested", "CPTAC-GBM-Clinical__Supratentorial Localization", "CPTAC-GBM-Clinical__Tumor Confined To Organ Of Origin",
    "CPTAC-GBM-Clinical__Tumor Focality", "CPTAC-GBM-Clinical__Tumor Regression Grade", "CPTAC-GBM-Clinical__Vascular Invasion Type",
    "CPTAC-GBM-Clinical__Wilms Tumor Histologic Subtype", "CPTAC-GBM-Clinical__Breslow Thickness", "CPTAC-GBM-Clinical__Gleason Grade Group",
    "CPTAC-GBM-Clinical__Igcccg Stage", "CPTAC-GBM-Clinical__International Prognostic Index", "CPTAC-GBM-Clinical__Largest Extrapelvic Peritoneal Focus",
    "CPTAC-GBM-Clinical__Masaoka Stage", "CPTAC-GBM-Clinical__Non Nodal Regional Disease", "CPTAC-GBM-Clinical__Non Nodal Tumor Deposits",
    "CPTAC-GBM-Clinical__Ovarian Specimen Status", "CPTAC-GBM-Clinical__Ovarian Surface Involvement", "CPTAC-GBM-Clinical__Percent Tumor Invasion",
    "CPTAC-GBM-Clinical__Peritoneal Fluid Cytological Status", "CPTAC-GBM-Clinical__Primary Gleason Grade", "CPTAC-GBM-Clinical__Secondary Gleason Grade",
    "CPTAC-GBM-Clinical__Weiss Assessment Score", "CPTAC-GBM-Clinical__Adrenal Hormone", "CPTAC-GBM-Clinical__Ann Arbor B Symptoms Described",
    "CPTAC-GBM-Clinical__Eln Risk Classification", "CPTAC-GBM-Clinical__Figo Staging Edition Year", "CPTAC-GBM-Clinical__Gleason Grade Tertiary",
    "CPTAC-GBM-Clinical__Gleason Patterns Percent", "CPTAC-GBM-Clinical__Margin Distance", "CPTAC-GBM-Clinical__Margins Involved Site",
    "CPTAC-GBM-Clinical__Pregnant At Diagnosis", "CPTAC-GBM-Clinical__Satellite Nodule Present", "CPTAC-GBM-Clinical__Sites Of Involvement",
    "CPTAC-GBM-Clinical__Tumor Depth", "CPTAC-GBM-Clinical__Who Cns Grade", "CPTAC-GBM-Clinical__Who Nte Grade",
    "CPTAC-GBM-Clinical__Diagnosis Uuid",
    "brats23_metadata_flattened__global__vasari_f22_ncet_crosses_midline",
    "brats23_metadata_flattened__global__vasari_f6_proportion_ncet",

    "CPTAC-GBM-Clinical__Classification of Tumor",
    "CPTAC-GBM-Clinical__AJCC Pathologic Stage",
    "CPTAC-GBM-Clinical__AJCC Staging System Edition",
    "UPENN-GBM_Clinical__Time_since_baseline_preop", # all zero
    "tcga-tcia__progression_or_recurrence", # Not Reported
    "tcga-tcia__tumor_grade", # Not Reported
    "tcga-tcia__last_known_disease_status", # Not Reported
    "tcga-tcia__morphology", # format issue (e.g. 3/1/9440  12:00:00 AM)
    "tcga-tcia__classification_of_tumor", # Not Reported,
    

]



def clean_merged(
    merged,
    cleaned_merged_save_path='./datasheets/cleaned_merged.csv'
):
    print(f'Cleaning merged sheets...')
    ## Clean up merged
    merged_clean = merged.drop(columns=cols_to_delete)
    cols_to_check = ["midline_summary__dataset", "id"]  

    mask = merged_clean[cols_to_check].map(
        lambda x: pd.isna(x) or str(x).strip() in ["", "--"]
    ).any(axis=1)

    merged_clean = merged_clean.loc[~mask].copy()
    merged_clean = merged_clean.set_index("id")
    merged_clean.to_csv(cleaned_merged_save_path, index=True)

    print(f'Datasheets cleaned!')
    print(f'{cleaned_merged_save_path} contains {merged_clean.shape[0]} rows and {merged_clean.shape[1]} columns.')
    print('First 10 keys', list(merged_clean.keys())[:10], '\n')




def main(args):

    # 1. Flatten brats23_metadata.json into CSV
    with open(args.brats_metadata, "r") as f:
        data = json.load(f)
    flattened = {sid: flatten_json(payload) for sid, payload in data.items()}
    df = pd.DataFrame.from_dict(flattened, orient="index").reset_index().rename(columns={"index": "id"})
    df.to_csv('./datasheets/brats23_metadata_flattened.csv')

    # 2. Merge all datasheet CSVs
    merged = merge_datasheets()

    if args.clean_merged:
        # 3. Clean merged CSV
        clean_merged(merged)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--brats_metadata", type=str, default="../llm_report_generation/brats23_metadata.json")
    p.add_argument("--clean_merged", type=bool, default=True)
    args = p.parse_args()
    main(args)