import os
from tqdm import tqdm
import re
import random
random.seed(42)

from omegaconf import OmegaConf
from typing import List
from collections import defaultdict
import numpy as np
import pandas as pd
import json

import cv2
import openslide
from PIL import Image


class HancockPatient:
    """
    Class to handle the data of a single patient in the Hancock dataset.
    Args:
        patient_id (str): The ID of the patient in format "XXX" (3 digits)
    """

    def __init__(self, patient_id: str) -> None:
        conf = OmegaConf.load("neurips25/configs/base.yaml")
        self.archive_path = conf.hancock.archive_path
        self.extract_path = conf.hancock.extract_path
        self.thumbnails_path = conf.hancock.thumbnails_path

        self.patient_id = patient_id
        if len(self.patient_id) < 3:
            self.patient_id = str(self.patient_id).zfill(3)

        # Load clinical data
        clinical_data_path = os.path.join(self.extract_path, "StructuredData", "clinical_data.json")
        with open(clinical_data_path, 'r') as f:
            clinical_data = json.load(f)
        clinical_data_df = pd.DataFrame(clinical_data)
        clinical_data_df = clinical_data_df[clinical_data_df["patient_id"] == self.patient_id].set_index("patient_id")

        # Load pathological data
        pathological_data_path = os.path.join(self.extract_path, "StructuredData", "pathological_data.json")
        with open(pathological_data_path, 'r') as f:
            pathological_data = json.load(f)
        pathological_data_df = pd.DataFrame(pathological_data)
        pathological_data_df = pathological_data_df[pathological_data_df["patient_id"] == self.patient_id].set_index("patient_id")

        # Load blood data
        blood_data_path = os.path.join(self.extract_path, "StructuredData", "blood_data.json")
        with open(blood_data_path, 'r') as f:
            blood_data = json.load(f)
        blood_data_df = pd.DataFrame(blood_data)
        blood_data_df = blood_data_df[blood_data_df["patient_id"] == self.patient_id]
        if len(blood_data_df) == 0:
            print("No blood data found for patient:", self.patient_id)
            blood_data_df_with_ref_ranges = None
        else:
            # Load blood data reference ranges
            blood_data_reference_ranges_path = os.path.join(self.extract_path, "StructuredData", "blood_data_reference_ranges.json")
            with open(blood_data_reference_ranges_path, 'r') as f:
                blood_data_reference_ranges = json.load(f)
            blood_data_reference_ranges_df = pd.DataFrame(blood_data_reference_ranges)

            # join blood data with reference ranges (specifying too low / too high)
            def value_status(row):
                sex = row["sex"]
                if sex == "male":
                    min_col = "normal_male_min"
                    max_col = "normal_male_max"
                elif sex == "female":
                    min_col = "normal_female_min"
                    max_col = "normal_female_max"
                else:
                    raise ValueError("Unexpected sex value")

                too_low = False
                too_high = False
                in_range = False

                if pd.isna(row[min_col]) and pd.isna(row[max_col]):
                    in_range = True
                elif pd.isna(row[min_col]):
                    too_high = row["value"] > row[max_col]
                    in_range = not too_high
                elif pd.isna(row[max_col]):
                    too_low = row["value"] < row[min_col]
                    in_range = not too_low
                else:
                    too_low = row["value"] < row[min_col]
                    too_high = row["value"] > row[max_col]
                    in_range = not too_low and not too_high

                return {"too_low": too_low, "too_high": too_high, "in_range": in_range}

            blood_data_df_with_ref_ranges = blood_data_df.merge(
                blood_data_reference_ranges_df,
                on=["group", "LOINC_name", "unit", "analyte_name"],
                how="left"
            )
            blood_data_df_with_ref_ranges["sex"] = blood_data_df_with_ref_ranges.apply(
                lambda row: clinical_data_df.loc[row["patient_id"]]["sex"] if pd.notna(row["patient_id"]) else None,
                axis=1
            )

            status_df = blood_data_df_with_ref_ranges.apply(
                lambda row: value_status(row) if pd.notna(row["value"]) else {"too_low": False, "too_high": False, "in_range": True},
                axis=1,
                result_type='expand'
            )

            blood_data_df_with_ref_ranges[["too_low", "too_high", "in_range"]] = status_df
            blood_data_df_with_ref_ranges["out_of_range"] = blood_data_df_with_ref_ranges["too_low"] | blood_data_df_with_ref_ranges["too_high"]


        self.clinical_data_df = clinical_data_df
        self.pathological_data_df = pathological_data_df
        self.blood_data_df_with_ref_ranges = blood_data_df_with_ref_ranges

        # Load history
        history_text_path = os.path.join(self.extract_path, "TextData", "histories_english", f"SurgeryReport_History_{self.patient_id}.txt")
        if os.path.exists(history_text_path):
            with open(history_text_path, 'r') as f:
                self.history_text = f.read()
        else:
            self.history_text = None
            print("No history text found for patient:", self.patient_id)

        def parse_ICD_OPS_codes(icd_codes_text: str) -> list[str]:
            codes = icd_codes_text.split("]")
            result = []
            for code in codes:
                code = code.strip()
                if code:
                    code = code.split("[")
                    if len(code) != 2:
                        print("Error parsing ICD code:", code)
                        continue
                    icd_desc, icd_code = code[0], code[1]
                    icd_desc = icd_desc.strip()
                    icd_code = icd_code.strip()
                    result.append((icd_desc, icd_code))
            return result
        
        icd_codes_path = os.path.join(self.extract_path, "TextData", "icd_codes", f"SurgeryReport_ICD_Codes_{self.patient_id}.txt")
        if os.path.exists(icd_codes_path):
            with open(icd_codes_path, 'r') as f:
                icd_codes_text = f.read()
                self.icd_codes = parse_ICD_OPS_codes(icd_codes_text)
        else:
            self.icd_codes = None
            print("No ICD codes text found for patient:", self.patient_id)
        
        ops_codes_path = os.path.join(self.extract_path, "TextData", "ops_codes", f"SurgeryReports_OPS_Codes_{self.patient_id}.txt")
        if os.path.exists(ops_codes_path):
            with open(ops_codes_path, 'r') as f:
                ops_codes_text = f.read()
                self.ops_codes = parse_ICD_OPS_codes(ops_codes_text)
        else:
            self.ops_codes = None
            print("No OPS codes text found for patient:", self.patient_id)

        # Load surgery report
        surgery_report_path = os.path.join(self.extract_path, "TextData", "reports_english", f"SurgeryReport_{self.patient_id}.txt")
        if os.path.exists(surgery_report_path):
            with open(surgery_report_path, 'r') as f:
                self.surgery_report_text = f.read()
        else:
            self.surgery_report_text = None
            print("No surgery report text found for patient:", self.patient_id)

        # Load surgery descriptions
        surgery_descriptions_path = os.path.join(self.extract_path, "TextData", "surgery_descriptions_english", f"SurgeryDescriptionEnglish_{self.patient_id}.txt")
        if os.path.exists(surgery_descriptions_path):
            with open(surgery_descriptions_path, 'r') as f:
                self.surgery_descriptions_text = f.read()
        else:
            self.surgery_descriptions_text = None
            print("No surgery descriptions text found for patient:", self.patient_id)

        # Load IHC positive cell density measurements
        # first get patient's block nb
        patient_cores = self.get_patient_cores_coordinates()
        patient_cores = {os.path.basename(k): v for k, v in patient_cores.items()}
        blocks = set()
        for k, v in patient_cores.items():
            block = re.search(r"_(block\d+)", k)
            block = block.group(1) if block else None
            if block:
                if block not in blocks:
                    blocks.add(block)
        assert self.patient_id in ["732"] or len(blocks) == 1, f"Found {len(blocks)} blocks for patient {self.patient_id}: {blocks}. Check if the patient is really in multiple blocks."
        self.block_number = blocks.pop() if len(blocks) == 1 else list(blocks) # normally each patient is only in one block, but there are exceptions
        blocks = [self.block_number] if isinstance(self.block_number, str) else self.block_number
        # then collect TMA measurements for all views x markers
        self.tma_measurements = []
        for block in blocks:
            for view in ["TumorCenter", "InvasionFront"]:
                for marker in ["CD3", "CD8", "CD56", "CD68", "CD163", "MHC1", "PDL1"]: # HE is not included
                    view_marker_block_celldensity_measurements = os.path.join(self.extract_path, "TMA_CellDensityMeasurements_recomputed", f"{view}_{marker}_{block}.csv")
                    if os.path.exists(view_marker_block_celldensity_measurements):
                        patient_cell_measurements = pd.read_csv(view_marker_block_celldensity_measurements, sep="\t")
                        patient_cell_measurements = patient_cell_measurements[patient_cell_measurements["Case ID"] == int(self.patient_id)]
                        for i, row in patient_cell_measurements.iterrows():
                            image_position, marker, _ = row["Image"].split("_")
                            self.tma_measurements.append({
                                "image_position": image_position,
                                "marker": marker,
                                "num_detected_cells": row["Num Detections"],
                                "num_marker_positive_cells": row["Num Positive"],
                                "num_marker_negative_cells": row["Num Negative"],
                                "num_marker_positive_cells_percent": row["Positive %"],
                                "num_positive_per_mm2": row["Num Positive per mm^2"],
                            })
                    else:
                        print(f"File {view_marker_block_celldensity_measurements} not found. Skipping.")
                        continue

    
    def has_one_abnormal_blood_test(self) -> bool:
        """
        Returns true if at least one blood test is out of reference range
        """
        if self.blood_data_df_with_ref_ranges is None:
            return False
        return self.blood_data_df_with_ref_ranges["out_of_range"].any()
    
    def num_abnormal_blood_tests(self) -> int:
        """
        Returns the number of blood tests that are out of reference range
        """
        if self.blood_data_df_with_ref_ranges is None:
            return 0
        return self.blood_data_df_with_ref_ranges["out_of_range"].sum()
    
    def num_blood_tests(self) -> int:
        """
        Returns the total number of blood tests
        """
        if self.blood_data_df_with_ref_ranges is None:
            return 0
        return len(self.blood_data_df_with_ref_ranges)
    
    def to_json(self, abnormal_blood_data_only=True, compact=True) -> dict:
        """
        Get all data for the patient in a dictionary format
        Args:
            abnormal_blood_data_only (bool): If true, only the blood tests that are out of reference range are returned (this can significantly reduce the size of the json file)
        Returns:
            dict: Dictionary with all data for the patient
        """
        result = dict()
        result["patient_id"] = self.patient_id
        result["clinical_data"] = self.clinical_data_df.iloc[0].to_dict()
        result["pathological_data"] = self.pathological_data_df.iloc[0].to_dict()
        if self.blood_data_df_with_ref_ranges is None:
            result["blood_data"] = None
        else:
            blood_data_col_drop = ["patient_id", "normal_male_min", "normal_male_max", "normal_female_min", "normal_female_max", "sex", "in_range"]
            if abnormal_blood_data_only:
                result["blood_data"] = self.blood_data_df_with_ref_ranges.drop(blood_data_col_drop,axis=1)[self.blood_data_df_with_ref_ranges["out_of_range"] == True].to_dict(orient="records")
            else:
                result["blood_data"] = self.blood_data_df_with_ref_ranges.drop(blood_data_col_drop,axis=1).to_dict(orient="records")
            if compact:
                datas = result["blood_data"].copy()
                result["blood_data"] = ""
                for data in datas:
                    result["blood_data"] += f"Blood test for {data['analyte_name']} ({data['LOINC_name']}, {data['unit']}) is {'too low' if data['too_low'] else 'too high' if data["too_high"] else "normal"} ({data['value']})\n"
        if compact:
            seen = set()
            result["ihc_measurements"] = ""
            for measurement in self.tma_measurements:
                if not np.isnan(measurement['num_positive_per_mm2']):
                    if f"{measurement['image_position']}_{measurement['marker']}" in seen:
                        result["ihc_measurements"] += f"Another "
                    else:
                        seen.add(f"{measurement['image_position']}_{measurement['marker']}")
                    result["ihc_measurements"] += f"{measurement['image_position']} image with {measurement['marker']} marker has {measurement['num_detected_cells']} detected cells of which {measurement['num_marker_positive_cells_percent']}% are positive ({measurement['num_positive_per_mm2']}/mm^2)\n"
        else:
            result["ihc_measurements"] = self.tma_measurements
        result["history_text"] = self.history_text
        result["icd_codes"] = self.icd_codes
        result["ops_codes"] = self.ops_codes
        result["surgery_report_text"] = self.surgery_report_text
        result["surgery_descriptions_text"] = self.surgery_descriptions_text
        return result

    def get_patient_cores_coordinates(self) -> dict:
        """
        Reads CSV files from a directory and maps patient IDs to their corresponding core coordinates.
        Args:
            none
        Returns:
            dict: A dictionary mapping SVS file names to lists of core coordinates for the given patient.
        """
        if hasattr(self, "patient_cores"):
            return self.patient_cores
        self.patient_cores = {}
        csv_dir = os.path.join(self.extract_path, "TMA_Maps","TMA_Maps")
        for filename in os.listdir(csv_dir):
            if filename.endswith(".csv"):
                csv_path = os.path.join(csv_dir, filename)
                df = pd.read_csv(csv_path)
                for tumor_pos in ["TMA_TumorCenter", "TMA_InvasionFront"]:
                # for tumor_pos in ["TMA_TumorCenter"]:
                    for stain in ["CD3", "CD8", "CD56", "CD68", "CD163", "HE", "MHC1", "PDL1"]:
                        svs_file = filename.replace("TMA_Map_", f"TumorCenter_{stain}_").replace(".csv", ".svs") if tumor_pos == "TMA_TumorCenter" else filename.replace("TMA_Map_", f"InvasionFront_{stain}_").replace(".csv", ".svs")
                        svs_file = os.path.join(self.extract_path, tumor_pos, tumor_pos, stain, svs_file)
                        self.patient_cores[svs_file] = df[df["Case ID"] == int(self.patient_id)]["core"].tolist()
        return {k: v for k, v in self.patient_cores.items() if v}

    def extract_precomputed_patient_cores(self, path: str = "TMA_Cores") -> List[Image]:
        """
        Extracts and saves images of cores belonging to a specific patient from precomputed SVS files.

        Args:
            none
        Returns:
            list[Image]: A list of extracted core images.
        """
        files = []
        for stain in ["CD3", "CD8", "CD56", "CD68", "CD163", "HE", "MHC1", "PDL1"]:
            for file in os.listdir(os.path.join(self.extract_path, path, f"tma_tumorcenter_{stain}")):
                if file.endswith(".png") and f"patient{self.patient_id}" in file:
                    files.append(os.path.join(self.extract_path, path, f"tma_tumorcenter_{stain}", file))
        if not files:
            print(f"No cores found for patient {self.patient_id}")
            return []
        print(f"Found {len(files)} cores for patient {self.patient_id}")
        assert len(files) < 70
        # read png images
        cores = defaultdict(lambda: defaultdict(list))
        for file_name in tqdm(files, desc="Extracting cores", unit="core"):
            core = cv2.imread(file_name)
            if core is None:
                print(f"Error reading {file_name}")
                continue
            view,stain,block,x,y,patient_id = file_name.split("/")[-1].split("_")
            cores[view][stain].append((core, block, (x,y), patient_id))
        return cores


    def _extract_cores_from_centroids(self, slide_path: str, array_position: str, centroid_um_x: float, centroid_um_y: float, level: int = 0, downsample: int = 1) -> Image:
        """
        Extracts a core from a TMA slide using the given centroid coordinates.
        Args:
            slide_path (str): The path to the TMA slide.
            array_position (str): The position of the TMA core in the array.
            centroid_um_x (float): The x-coordinate of the centroid in micrometers.
            centroid_um_y (float): The y-coordinate of the centroid in micrometers.
        Returns:
            Image: The extracted core image.
        """
        slide = openslide.OpenSlide(slide_path)

        assert slide.properties['aperio.MPP'] == slide.properties['openslide.mpp-x'] == slide.properties['openslide.mpp-y']
        mpp = float(slide.properties['openslide.mpp-x'])
        core_diameter_um = 2200

        base_width, base_height = slide.dimensions

        # === Convert µm to base-level pixels ===
        x_px = int(centroid_um_x / mpp)
        y_px = int(centroid_um_y / mpp)
        core_diameter_px = int(core_diameter_um / mpp)

        # === Adjust for 180° rotation ===
        x_px = base_width - x_px
        y_px = base_height - y_px

        # === Compute top-left in base-level pixels ===
        top_left_x = x_px - core_diameter_px // 2
        top_left_y = y_px - core_diameter_px // 2

        # === Compute size at the desired level ===
        width_lvl = core_diameter_px // downsample
        height_lvl = core_diameter_px // downsample
        size = (width_lvl, height_lvl)

        region = slide.read_region((top_left_x, top_left_y), level, size)
        return region

    def extract_patient_cores_from_recomputed_centroids(self, level : int = 0, downsample : int = 1):
        """
        Extracts cores from recomputed centroids if the TMA_CellDensityMeasurements_recomputed folder exists and contains any TMA centroids for the patient.
        This function will extract the files if they do not already exist and return the paths.

        Args:
            level (int): The level of the slide to extract the core from.
            downsample (int): The downsample factor for the extracted core.
        Returns:
            paths (list[str]): A list of paths to the extracted/existing core images.
        """
        patient_cores = self.get_patient_cores_coordinates()
        patient_cores = {os.path.basename(k): v for k, v in patient_cores.items()}
        blocks = set()
        for k, v in patient_cores.items():
            if len(v) > 0:
                block = re.search(r"_(block\d+)", k)
                if block:
                    block = block.group(1)
                    if block not in blocks:
                        blocks.add(block)
        if len(blocks) > 1:
            print(f"Found {len(blocks)} blocks for patient {self.patient_id}. Cannot compute centroids.")
            print(patient_cores)
            return
        block = blocks.pop()
        if not patient_cores:
            print(f"No cores found for patient {self.patient_id}. Cannot compute centroids.")
            return
        recomputed_centroids_path = os.path.join(self.extract_path, "TMA_CellDensityMeasurements_recomputed")
        out_path = os.path.join(self.extract_path, "TMA_cores_recomputed", f"patient{self.patient_id}")
        os.makedirs(out_path, exist_ok=True)
        views = ["TumorCenter", "InvasionFront"]
        stains = ["CD3", "CD8", "CD56", "CD68", "CD163", "HE", "MHC1", "PDL1"]
        paths = []
        for view in views:
            for stain in stains:
                tma_celldensity_filename = f"{view}_{stain}_{block}.svs"
                if tma_celldensity_filename not in patient_cores:
                    print(f"File {tma_celldensity_filename} not found in patient cores. Skipping.")
                    continue
                tma_celldensity_measurements_path = os.path.join(recomputed_centroids_path, tma_celldensity_filename.replace(".svs", ".csv"))
                if os.path.exists(tma_celldensity_measurements_path):
                    # print(f"Found TMA centers for {tma_celldensity_filename}")
                    centers = pd.read_csv(tma_celldensity_measurements_path, sep="\t")
                    centers = centers[(centers["Case ID"] == int(self.patient_id)) & (centers["Image"] == tma_celldensity_filename)]
                    tumor_pos = "TMA_TumorCenter" if "TumorCenter" in tma_celldensity_filename else "TMA_InvasionFront"
                    slide_path = os.path.join(self.extract_path, tumor_pos, tumor_pos, stain, tma_celldensity_filename)
                    if len(centers) == 0:
                        print(f"No centers found inside csv for {tma_celldensity_filename}")
                        continue
                    for i, (_, row) in enumerate(centers.iterrows()):
                        centroid_um_x = row["Centroid X µm"]
                        centroid_um_y = row["Centroid Y µm"]
                        core_path = os.path.join(out_path, f"{self.patient_id}_{view}_{stain}_{i}_lvl{level}_dwnsmpl{downsample}.png")
                        if os.path.exists(core_path):
                            paths.append(core_path)
                            continue
                        # print(f"Extracting core for {view} {stain} at ({centroid_um_x}, {centroid_um_y})")
                        core = self._extract_cores_from_centroids(slide_path, tma_celldensity_filename, centroid_um_x, centroid_um_y, level=level, downsample=downsample)
                        # save image
                        core = np.array(core)
                        core = cv2.cvtColor(core, cv2.COLOR_RGBA2BGR)
                        # print("core shape = ", core.shape)
                        cv2.imwrite(core_path, core)
                        paths.append(core_path)
                else:
                    continue
        return paths

    def get_thumbnails(self, recomputed: bool = True) -> list[str]:
        """
        Gets low resolution thumbnails of the patient IHC images. Uses recomputed thumbnails if they exist.
        Args:
            recomputed (bool): If true, the recomputed thumbnails are used if they exist
        Returns:
            list[str]: A list of full paths to each thumbnail image
        """
        if recomputed:
            return self.extract_patient_cores_from_recomputed_centroids(level=2, downsample=16)
        patient_thumbnails_path = os.path.join(self.thumbnails_path, f"patient{self.patient_id}")
        if not os.path.exists(patient_thumbnails_path) or len(os.listdir(patient_thumbnails_path)) == 0:
            print("No thumbnails found for patient", self.patient_id)
            return []
        result = []
        for file_name in os.listdir(patient_thumbnails_path):
            if file_name.endswith(".jpg"):
                result.append(os.path.join(patient_thumbnails_path, file_name))
        return result

    def get_WSI(self, with_annotations: bool = True) -> list[str]:
        """
        Find the WSIs of the patient if any
        Args:
            with_annotations (bool): If true, the annotations are also returned if any
        Returns:
            list[str]: A list of full paths to each WSI image
        """
        paths = []
        WSI_dirs = [d for d in os.listdir(self.extract_path) if d.startswith("WSI_")]
        for dir_name in WSI_dirs:
            for root, _, files in os.walk(os.path.join(self.extract_path, dir_name)):
                for file_name in files:
                    if file_name.endswith(".svs") or (with_annotations and file_name.endswith("geojson")):
                        # check if the patient_id is in the file name
                        if self.patient_id in file_name:
                            paths.append(os.path.join(root, file_name))
        return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract patient data from Hancock dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient_id", type=int, help="Patient ID (range 1-763)")
    group.add_argument("--random", action="store_true", help="Test mode")
    args = parser.parse_args()

    # ------------------------------ get random patient ------------------------------------
    if args.random:
        conf = OmegaConf.load("neurips25/configs/base.yaml")
        extract_path = conf.hancock.extract_path
        clinical_data_path = os.path.join(extract_path, "StructuredData", "clinical_data.json")
        with open(clinical_data_path, 'r') as f:
            clinical_data = json.load(f)
        clinical_patients = pd.DataFrame(clinical_data)["patient_id"].to_list()
        patient_id = random.choice(clinical_patients)
        print(f"Random patient: {patient_id} out of {len(clinical_patients)} patients")
    else:
        patient_id = str(args.patient_id)
        if len(patient_id) < 3:
            patient_id = str(patient_id).zfill(3)
    # --------------------------------------------------------------------------------------

    # ------------------------------- get patient data ------------------------------------
    patient = HancockPatient(patient_id)
    # ---------------------------------------------------------------------------------------


    print("Patient data loaded.")
    print()
    print("has_abnormal_blood_test:", patient.has_one_abnormal_blood_test())
    print("num_abnormal_blood_tests:", patient.num_abnormal_blood_tests())
    print("num_blood_tests:", patient.num_blood_tests())
    print("Patient to json:")
    print(json.dumps(patient.to_json(), indent=2))
    print()
    print("WSI:", len(patient.get_WSI()), "available")
    print()
    print("IHC:", len(patient.get_thumbnails()), "thumbnails available for markers CD3, CD8, CD56, CD68, CD163, HE, MHC1, PDL1")
    print()
    print("TMA recomputed centroids:")
    paths = patient.extract_patient_cores_from_recomputed_centroids()
    print(f"Found {len(paths)} (recomputed) cores for patient {patient_id}")
    print()
    print("TMA pre-computed cores:")
    cores = patient.extract_precomputed_patient_cores()
    for view, stains in cores.items():
        print(f"View: {view}")
        for stain, cores in stains.items():
            print(f"  Stain: {stain}")
            for core, block, coords, patient_id in cores:
                print(f"    Core: {core.shape}, Block: {block}, Coords: {coords}, Patient ID: {patient_id}")
