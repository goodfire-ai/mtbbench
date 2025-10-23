import os
import json
from omegaconf import OmegaConf
from neurips25.utils.hancock_patient import HancockPatient
from openslide import OpenSlide
import geopandas as gpd
from shapely.affinity import scale
import matplotlib.pyplot as plt
import shutil
from PIL import Image


def plot_wsi(wsi_path, annotation_path):
    """
    Plot the WSI with the annotations overlayed.
    """
    wsi = OpenSlide(wsi_path)
    level = 2
    downsample = wsi.level_downsamples[level]
    dims = wsi.level_dimensions[level]
    region = wsi.read_region(location=(0, 0), level=level, size=dims).convert('RGB')

    roi_scaled = gpd.read_file(annotation_path)
    roi_scaled["geometry"] = roi_scaled["geometry"].apply(
        lambda geom: scale(geom, xfact=1/downsample, yfact=1/downsample, origin=(0, 0))
    )

    _, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(region)
    roi_scaled.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
    plt.axis("off")
    plt.show()


def extract_roi(wsi_path, annotation_path):
    """
    Extract the region of interest (ROI) from the WSI based on the annotation.
    """
    wsi = OpenSlide(wsi_path)
    annotation = gpd.read_file(annotation_path)

    bounds = annotation.total_bounds 

    level = 2
    downsample = wsi.level_downsamples[level]
    minx, miny, maxx, maxy = bounds
    x = int(minx / downsample)
    y = int(miny / downsample)
    width = int((maxx - minx) / downsample)
    height = int((maxy - miny) / downsample)

    # Add padding around the ROI
    padding = int(100 / downsample)
    x = max(0, x - padding)
    y = max(0, y - padding)
    width += 2 * padding
    height += 2 * padding

    # Read the patch from WSI
    patch = wsi.read_region(
        location=(int(x * downsample), int(y * downsample)),
        level=level,
        size=(width, height)
    ).convert('RGB')
    wsi.close()

    return patch


def resize_image(patch, max_side):
    """
    Resize the image patch to have a maximum side length of max_side.
    """
    width, height = patch.size

    if max(width, height) > max_side:
        if width > height:
            new_width = max_side
            new_height = int(height * (max_side / width))
        else:
            new_height = max_side
            new_width = int(width * (max_side / height))
        patch = patch.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return patch


def save_patient_case(patient_id, cases_path):
    patient = HancockPatient(patient_id)
    p = patient.to_json(abnormal_blood_data_only=False)

    patient_dir = os.path.join(cases_path, patient_id)
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)

    # Text data
    patient_clinical_data = p["clinical_data"]
    short_keys = ["year_of_initial_diagnosis", "age_at_initial_diagnosis", "sex", "smoking_status"]
    patient_clinical_data = {key: patient_clinical_data[key] for key in short_keys}
    with open(os.path.join(patient_dir, "patient_clinical_data.json"), 'w') as f:
        json.dump(patient_clinical_data, f)
    patient_pathological_data = p["pathological_data"]
    with open(os.path.join(patient_dir, "patient_pathological_data.json"), 'w') as f:
        json.dump(patient_pathological_data, f)
    patient_history_text = p["history_text"]
    with open(os.path.join(patient_dir, "history_text.txt"), 'w') as f:
        json.dump(patient_history_text, f)
    patient_icd_codes = p["icd_codes"]
    with open(os.path.join(patient_dir, "icd_codes.json"), 'w') as f:
        json.dump(patient_icd_codes, f)
    patient_ops_codes = p["ops_codes"]
    with open(os.path.join(patient_dir, "ops_codes.json"), 'w') as f:
        json.dump(patient_ops_codes, f)
    patient_surgery_report_text = p["surgery_report_text"]
    with open(os.path.join(patient_dir, "surgery_report.txt"), 'w') as f:
        json.dump(patient_surgery_report_text, f)
    patient_surgery_descriptions_text = p["surgery_descriptions_text"]
    with open(os.path.join(patient_dir, "surgery_descriptions.txt"), 'w') as f:
        json.dump(patient_surgery_descriptions_text, f)
    patient_tma_measurements = p["ihc_measurements"]
    with open(os.path.join(patient_dir, "patient_tma_measurements.txt"), 'w') as f:
        json.dump(patient_tma_measurements, f)
    patient_blood_data = patient.to_json(abnormal_blood_data_only=False, compact=False)["blood_data"]
    with open(os.path.join(patient_dir, "patient_blood_data.json"), 'w') as f:
        json.dump(patient_blood_data, f)
    patient_blood_data = p["blood_data"]
    with open(os.path.join(patient_dir, "patient_blood_data_for_question_generation.txt"), 'w') as f:
        json.dump(patient_blood_data, f)
    blood_data_reference_ranges_path = os.path.join(extract_path, "StructuredData", "blood_data_reference_ranges.json")
    shutil.copy(blood_data_reference_ranges_path, os.path.join(patient_dir, "blood_data_reference_ranges.json"))

    # Export WSI
    patient_wsis = patient.get_WSI()
    grouped_wsi = {}
    for wsi_path in patient_wsis:
        file_name = os.path.basename(os.path.splitext(wsi_path)[0])
        if file_name not in grouped_wsi:
            grouped_wsi[file_name] = []
        grouped_wsi[file_name].append(wsi_path)

    # Each group is WSI and annotation or only WSI
    for group in grouped_wsi:
        for g in grouped_wsi[group]:
            if g.endswith(".svs"):
                wsi_path = g
            else:
                annotation_path = g
        
        wsi = OpenSlide(wsi_path)
        overview_level = 2
        overview_dims = wsi.level_dimensions[overview_level]
        overview_image = wsi.read_region((0, 0), overview_level, overview_dims).convert('RGB')
        overview_image = resize_image(overview_image, 1500)
        overview_image.save(os.path.join(patient_dir, f"{group}.jpg"), quality=95)

        if annotation_path:
            patch = extract_roi(wsi_path, annotation_path)
            patch = resize_image(patch, 1500)
            patch.save(os.path.join(patient_dir, f"{group}_roi.jpg"), quality=95)

        wsi.close()
        annotation_path = None

    # Export thumbnails    
    for thumbnails_path in patient.get_thumbnails():
        basename = os.path.basename(thumbnails_path)
        basename = "TMA_IHC_" + "_".join(basename.split("_")[1:-2]) + ".png"
        # Resize the image to max side length of 700 pixels
        img = Image.open(thumbnails_path)
        img = resize_image(img, 900)
        img.save(os.path.join(patient_dir, basename), quality=95)


if __name__ == "__main__":
    conf = OmegaConf.load("neurips25/configs/base.yaml")
    archive_path = conf.hancock.archive_path
    extract_path = conf.hancock.extract_path
    thumbnails_path = conf.hancock.thumbnails_path
    cases_path = conf.hancock.cases_path

    for patient_id in [296, 741, 476, 162, 583, 564, 334, 176, 121, 698, 403, 761, 706, 740, 664, 346, 530, 120, 342, 225, 116, 559, 632, 104, 723, 250, 606]:
        patient_id = str(patient_id)
        if len(patient_id) == 2:
            patient_id = "0" + patient_id
        if len(patient_id) == 1:
            patient_id = "00" + patient_id
        print(patient_id)
        save_patient_case(patient_id, cases_path)
