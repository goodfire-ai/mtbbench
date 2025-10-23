from omegaconf import OmegaConf
import os
import pandas as pd
import json
from tqdm import tqdm
import random
random.seed(42)

def convert_days_to_years(df: pd.DataFrame):
    if "AGE_SEQUENCING" in df.columns:
        df["AGE_SEQUENCING"] = round(df["AGE_SEQUENCING"] / 365, 4)
        df = df.sort_values("AGE_SEQUENCING")

    if "START_DATE" in df.columns:
        df["START_DATE"] = round(df["START_DATE"] / 365, 4)
        df = df.rename(columns={"START_DATE": "START_AGE"})
        df = df.sort_values("START_AGE")
    
    if "STOP_DATE" in df.columns:
        df["STOP_DATE"] = round(df["STOP_DATE"] / 365, 4)
        df = df.rename(columns={"STOP_DATE": "STOP_AGE"})
    return df


class Sample:

    def __init__(self, sample_id: str) -> None:
        conf = OmegaConf.load("neurips25/configs/base.yaml")
        self.metadata_path = conf.msk.metadata_path

        self.sample_id = sample_id

        # Load sample data
        clinical_sample = pd.read_csv(os.path.join(self.metadata_path, "data_clinical_sample.txt"))
        clinical_sample = convert_days_to_years(clinical_sample)
        self.clinical_sample = clinical_sample[clinical_sample["SAMPLE_ID"] == self.sample_id].dropna(axis=1)

        # Load mutation, CNA, SV data
        cna_data_chord = pd.read_csv(os.path.join(self.metadata_path, "data_cna.txt"))
        cna_data_chord = convert_days_to_years(cna_data_chord)
        mutations_chord = pd.read_csv(os.path.join(self.metadata_path, "data_mutations.txt"))
        mutations_chord = convert_days_to_years(mutations_chord)
        sv_data_chord = pd.read_csv(os.path.join(self.metadata_path, "data_sv.txt"))
        sv_data_chord = convert_days_to_years(sv_data_chord)
        self.cna = cna_data_chord[cna_data_chord["SAMPLE_ID"] == self.sample_id].dropna(axis=1)

        mutations_chord = mutations_chord.rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
        sv_data_chord.columns = map(str.upper, sv_data_chord.columns)
        self.mutation = mutations_chord[mutations_chord["SAMPLE_ID"] == self.sample_id].dropna(axis=1)
        self.sv = sv_data_chord[sv_data_chord["SAMPLE_ID"] == self.sample_id].dropna(axis=1)


    def to_json(self) -> dict:
        result = dict()
        result["sample_id"] = self.sample_id
        result["clinical_sample"] = self.clinical_sample.to_json(orient="records")

        result["cna"] = self.cna["Hugo_Symbol"].to_json(orient="records")
        result["mutation"] = self.mutation["Hugo_Symbol"].to_json(orient="records")
        if "SITE2_HUGO_SYMBOL" in self.sv.columns:
            result["sv"] = self.sv[["SITE1_HUGO_SYMBOL", "SITE2_HUGO_SYMBOL"]].to_json(orient="records")
        else:
            result["sv"] = self.sv["SITE1_HUGO_SYMBOL"].to_json(orient="records")

        return result
    

    def to_markdown(self) -> str:
        result = f"### Sample {self.sample_id}\n"
        result += "#### Clinical information\n"
        result += self.clinical_sample.to_markdown(index=False)
        return result



class Patient:
    # Notes for the Future: Consider passing the clinical files in the constructor as a dictionary or something
    # in order to save waiting time for loading pandas files

    def __init__(self, patient_id: str) -> None:
        conf = OmegaConf.load("neurips25/configs/base.yaml")
        self.metadata_path = conf.msk.metadata_path

        self.patient_id = patient_id

        # Load clinical patient data
        clinical_patient = pd.read_csv(os.path.join(self.metadata_path, "data_clinical_patient.txt"))
        clinical_patient = convert_days_to_years(clinical_patient)
        self.clinical_patient = clinical_patient[clinical_patient["PATIENT_ID"] == self.patient_id].dropna(axis=1)

        # Load clinical sample data
        clinical_sample = pd.read_csv(os.path.join(self.metadata_path, "data_clinical_sample.txt"))
        clinical_sample = convert_days_to_years(clinical_sample)
        sample_ids = clinical_sample[clinical_sample["PATIENT_ID"] == self.patient_id]["SAMPLE_ID"].to_list()
        self.samples = [Sample(sample_id) for sample_id in sample_ids]
        
        # Load all timeline data
        timeline_dataframes = dict()
        timeline_files = ["diagnosis", "surgery", "ca_15-3_labs", "ca_19-9_labs", "cancer_presence", "cea_labs", "gleason", "mmr", "pdl1",
                        "performance_status", "prior_meds", "progression", "psa_labs", "radiation", "specimen_surgery",
                        "specimen", "treatment", "tumor_sites"]
        for timeline_file in timeline_files:
            timeline_dataframes[timeline_file] = pd.read_csv(os.path.join(self.metadata_path, f"data_timeline_{timeline_file}.txt"))
            timeline_dataframes[timeline_file] = convert_days_to_years(timeline_dataframes[timeline_file])
            timeline_dataframes[timeline_file] = timeline_dataframes[timeline_file][timeline_dataframes[timeline_file]["PATIENT_ID"] == self.patient_id].dropna(axis=1)
            # Drop columns we already know
            columns_to_drop = ["PATIENT_ID", "EVENT_TYPE"]
            timeline_dataframes[timeline_file] = timeline_dataframes[timeline_file].drop(columns_to_drop, axis=1)

        # Group treatment dataframes
        self.treatment = dict()
        self.treatment["radiation"] = timeline_dataframes["radiation"]
        self.treatment["prior_meds"] = timeline_dataframes["prior_meds"]
        self.treatment["treatment"] = timeline_dataframes["treatment"]

        # Group diagnosis dataframes
        self.diagnosis = dict()
        self.diagnosis["diagnosis"] = timeline_dataframes["diagnosis"]
        self.diagnosis["cancer_presence"] = timeline_dataframes["cancer_presence"].drop(["STYLE_COLOR", "NLP_HAS_CANCER_PROBABILITY"], axis=1)
        self.diagnosis["performance_status"] = timeline_dataframes["performance_status"].drop(["STYLE_COLOR", "SUBTYPE"], axis=1)
        self.diagnosis["progression"] = timeline_dataframes["progression"].drop(["STYLE_COLOR", "NLP_PROGRESSION_PROBABILITY", "SUBTYPE"], axis=1)
        self.diagnosis["tumor_sites"] = timeline_dataframes["tumor_sites"].drop(["SUBTYPE"], axis=1)

        # Group pathology dataframes
        self.pathology = dict()
        self.pathology["gleason"] = timeline_dataframes["gleason"]
        self.pathology["mmr"] = timeline_dataframes["mmr"]
        self.pathology["pdl1"] = timeline_dataframes["pdl1"]

        # Group lab test dataframes
        self.labtest = dict()
        self.labtest["ca_15-3_labs"] = timeline_dataframes["ca_15-3_labs"].drop(["TEST"], axis=1)
        self.labtest["ca_19-9_labs"] = timeline_dataframes["ca_19-9_labs"].drop(["TEST"], axis=1)
        self.labtest["cea_labs"] = timeline_dataframes["cea_labs"].drop(["TEST"], axis=1)
        self.labtest["psa_labs"] = timeline_dataframes["psa_labs"].drop(["TEST"], axis=1)

        # Set individual attributes
        self.surgery = timeline_dataframes["surgery"]
        self.specimen_surgery = timeline_dataframes["specimen_surgery"]
        self.specimen = timeline_dataframes["specimen"]
    

    def to_json(self) -> dict:
        result = dict()
        result["patient_id"] = self.patient_id
        result["clinical_patient"] = self.clinical_patient.to_json(orient="records")
        result["samples"] = [sample.to_json() for sample in self.samples]

        result["surgery"] = self.surgery.to_json(orient="records")
        result["specimen_surgery"] = self.specimen_surgery.to_json(orient="records")
        result["specimen"] = self.specimen.to_json(orient="records")

        result["treatment"] = dict(map(lambda x: (x[0], x[1].to_json(orient="records")), self.treatment.items()))
        result["diagnosis"] = dict(map(lambda x: (x[0], x[1].to_json(orient="records")), self.diagnosis.items()))
        result["pathology"] = dict(map(lambda x: (x[0], x[1].to_json(orient="records")), self.pathology.items()))
        result["labtest"] = dict(map(lambda x: (x[0], x[1].to_json(orient="records")), self.labtest.items()))
        return result
    
    def collect_events_with_start_age(self, data, category_path=None):
        """
        Recursively collects all events with a START_AGE field in the nested JSON data.
        """
        if category_path is None:
            category_path = []

        events = []

        if isinstance(data, list):
            for item in data:
                events.extend(self.collect_events_with_start_age(item, category_path))
        elif isinstance(data, dict):
            for key, value in data.items():
                # Try parsing stringified JSON fields (if any)
                if isinstance(value, str):
                    try:
                        parsed_value = json.loads(value)
                        events.extend(self.collect_events_with_start_age(parsed_value, category_path + [key]))
                    except (json.JSONDecodeError, TypeError):
                        continue
                elif isinstance(value, (dict, list)):
                    events.extend(self.collect_events_with_start_age(value, category_path + [key]))
            
            if 'START_AGE' in data:
                event = data.copy()
                event['_category_path'] = category_path
                events.append(event)

        return events

    @staticmethod
    def sort_events_by_start_age(events):
        """
        Sort the list of events by START_AGE.
        """
        return sorted(events, key=lambda x: x['START_AGE'])

    @staticmethod
    def format(event):
        class default_id_dict(dict):
            @staticmethod
            def __missing__(key):
                return key
        def default_format(event):
            return ''.join(
                [f"{k}: {v.strip() if type(v)==str else v}, " for k,v in event.items() if k not in ['_category_path', 'START_AGE']]
            )
        d = default_id_dict({'0': 'No', '1': 'Yes', 'True': 'Yes', 'False': 'No', 'Y': 'Yes', 'N': 'No', 'CT': 'CT scan'})
        category = event['_category_path']
        if category == ['labtest', 'cea_labs']:
            assert set(event.keys()).difference(['START_AGE', '_category_path']) == set(['RESULT', 'LR_UNIT_MEASURE']), f"Unexpected keys in event: {event.keys()}"
            return f"CEA: {event['RESULT']:.4f} {event['LR_UNIT_MEASURE']}"
        elif category == ['diagnosis', 'progression']:
            assert set(event.keys()).difference(['START_AGE', '_category_path']) == set(['PROGRESSION', 'PROCEDURE_TYPE']), f"Unexpected keys in event: {event.keys()}"
            if event['PROCEDURE_TYPE'] == 'CT':
                if event['PROGRESSION'] == 'Y':
                    return f"CT scan reveals cancer has progressed."
                return f"CT scan reveals cancer has NOT progressed."
            return default_format(event)
        elif category == ['diagnosis', 'tumor_sites']:
            return f"SOURCE: {event['SOURCE']}, {event['SOURCE_SPECIFIC']}, TUMOR_SITE: {event['TUMOR_SITE']}, " + ''.join([
                f"{k}: {d[str(event[k])]}, " for k in set(event.keys()).difference(['SOURCE', 'SOURCE_SPECIFIC', 'TUMOR_SITE', 'START_AGE', '_category_path'])
            ])
        elif category == ['diagnosis', 'cancer_presence']:
            return f"HAS CANCER: {d[event['HAS_CANCER']]}, SUBTYPE: {event['SUBTYPE']}, PROCEDURE_TYPE: {event['PROCEDURE_TYPE']}, " + ''.join([
                f"{k}: {d[str(event[k])]}, " for k in set(event.keys()).difference(['SOURCE', 'SOURCE_SPECIFIC', 'TUMOR_SITE', 'START_AGE', '_category_path'])
            ])
        else:
            return default_format(event)

    @staticmethod
    def summarize_events(events, test=False):
        """
        Prints a clean timeline of events
        """
        result = ""
        for event in events:
            event_formatted = Patient.format(event)
            if test:
                _ = f"AGE: {event['START_AGE']:.3f}, {' > '.join(event['_category_path'])} --> {event_formatted}"
            else:
                result += f"AGE: {event['START_AGE']:.3f}, {' > '.join(event['_category_path'])} --> {event_formatted}\n"
        return result

    def get_sorted_events(self, min_age: float = 0, max_age: float = 10000) -> list:
        """
        Collects all events with START_AGE and sorts them by START_AGE.
        """
        all_events = self.collect_events_with_start_age(self.to_json())
        sorted_events = self.sort_events_by_start_age(all_events)
        sorted_events = [event for event in sorted_events if min_age <= event['START_AGE'] <= max_age]
        return sorted_events
    
    @staticmethod
    def test():
        """
        Test printing the timelines of 10 random patients
        """
        conf = OmegaConf.load("neurips25/configs/base.yaml")
        metadata_path = conf.msk.metadata_path
        clinical_patient = pd.read_csv(os.path.join(metadata_path, "data_clinical_patient.txt"))["PATIENT_ID"].to_list()
        patient_ids = random.choices(clinical_patient, k=10)
        events = [Patient(patient_id).get_sorted_events() for patient_id in patient_ids]
        for event in events:
            Patient.summarize_events(event, test=True)

if __name__ == "__main__":

    # get random patient
    import random
    random.seed(42)
    conf = OmegaConf.load("neurips25/configs/base.yaml")
    metadata_path = conf.msk.metadata_path
    clinical_patient = pd.read_csv(os.path.join(metadata_path, "data_clinical_patient.txt"))["PATIENT_ID"].to_list()
    patient_id = random.choice(clinical_patient)
    print(f"Random patient: {patient_id} out of {len(clinical_patient)} patients")

    patient = Patient(patient_id)
    events = patient.get_sorted_events()
    print(patient.summarize_events(events))
    print("\n\n")
    events = patient.get_sorted_events(min_age=80.360, max_age=80.426)
    print(patient.summarize_events(events))
