import json
import re
import os
from omegaconf import OmegaConf
import openai
from neurips25.utils.hancock_patient import HancockPatient


CONTEXT_SYSTEM_PROMPT = "You are an expert oncologist able to generate multiple choice questions based on the information you are provided with."
context_user_prompt = lambda clincal_data, history_text: f"You are provided with the following patient information: Clinical data {clincal_data} History text {history_text} Your task is to write a single paragraph clinical summary containing only the information available at the time of the initial presentation and diagnostic workup, prior to any confirmed diagnosis, pathological analysis, surgical intervention, or cancer staging. Do not include any histological findings, cancer-related terminology (e.g., carcinoma, malignancy, tumor), staging or grading details, or mention of surgeries that have already been performed. You may mention demographic information (e.g., age, sex), clinical symptoms, relevant lifestyle factors (e.g., smoking status), and any diagnostic procedures planned or underway before diagnosis (e.g., imaging, panendoscopy, biopsy). You must mention that a sample has been taken and H&E and IHC stainings have been made. The summary must be written as if it were documented before any diagnosis was confirmed."

HE_IHC_SYSTEM_PROMPT = "You are an expert oncologist able to generate multiple choice questions based on the information you are provided with."
he_ihc_user_prompt = lambda patient_tma_measurements, patient_pathological_data, patient_icd_codes: f"""
You are provided with the following data: 
TMA measurements {patient_tma_measurements} 
Pathological data {patient_pathological_data} 
ICD codes {patient_icd_codes} 

Guidelines:
- Questions must begin with [QUESTION] tag and answers with [ANSWER], such that they are easy to extract using regex.
- Write multiple-choice questions (exactly 2 or 6 choices, but ONLY 1 correct), showing also the correct one, all based ONLY on the information provided to you explicitly.
- The answer choices MUST follow the format:\nA) answer1\nB) answer2\nC) answer3\nD) answer4\nE) answer5\nF) answer6\n
- There must be only a single unambiguous correct answer. Do NOT include "Unable to determine" or "Not enough information" as an answer choice.
- Write 2 questions that require examining only H&E images, and feel free to use the pathological information - the user will not have access to it and will need to answer solely based on H&E images. Examples: What is the main cancer type according to this histopathology image?, Is there any evidence of vascular invasion on the WSIs from the primary site and lymph node? Etc.
- Write 4 questions that can be answered using only IHC images (CD3, CD8), and feel free to use the TMA measurements to guide what to focus on, such as expected regional differences in immune cell infiltration—but the user must answer based solely on IHC image review. Examples: What can we conclude about M2 macrophages based on the given IHC slides? Is there a significant T-cell infiltration in the invasion front?
- Write 3 questions that require examining both H&E and IHC images, using the TMA measurements and pathological information to guide you. The questions MUST require the user to analyze both H&E and IHC images. The question must not be answerable without having access to both H&E and IHC images. The incorrect answer choices must also stem from the TMA measurements and pathological information.
- Use the TMA measurements to guide the questions, but do not include their exact values in the questions or answers.
- Avoid simple recall-based questions or those that could be answered without image access.
- Do not include greeting text

Make the questions diverse and diagnostic in nature. Prefer questions that would be asked by a trained pathologist or for use in a tumor board setting.
"""

BLOOD_SYSTEM_PROMPT = "You are an expert oncologist able to generate multiple choice questions based on the information you are provided with."
blood_user_prompt = lambda patient_clinical_data, patient_pathological_data, patient_blood_data, blood_data_reference_ranges: f"""
You are provided with the following patient information: 

Clinical data {patient_clinical_data} 
Pathological data {patient_pathological_data} 
Blood data {patient_blood_data}
Blood data reference ranges {blood_data_reference_ranges}

Guidelines:
- The patient is about to undergo surgery. You MUST start by writing some context using [CONTEXT] tag about the incoming surgery and the blood tests taken related to it. 
- The context MUST not reveal anything about the actual contents of the blood tests. 
- Do not include patient demographic information. 
- Next, write 4 multiple-choice questions (exactly 6 choices, but ONLY 1 correct) related to the blood tests performed. 
- The questions MUST use the provided blood information and MUST NOT be answerable without having access to the blood tests. Example: Is there any renal dysfunction that could complicate chemotherapy?
- The questions MUST rely on the low/normal/high values of the blood tests and MUST requre the user to analyze them to answer the questions.
- Do not reveal the actual blood test values in the questions, so ONLY a person WITH ACCESS to the blood tests can answer them!
- The incorrect answer choices MUST NOT be obvious and MUST also be generated based on the blood tests. I.e. an incorrect answer choice MUST NOT be identifiable as incorrect without having access to the blood tests.
- Begin each question using [QUESTION] tag and specify the correct answer using [ANSWER] tag.
- The answer choices MUST follow the format:\nA) answer1\nB) answer2\nC) answer3\nD) answer4\nE) answer5\nF) answer6\n
- There must be only a single unambiguous correct answer. Do NOT include "Unable to determine" or "Not enough information" as an answer choice.
- Do not include greeting text

Make the questions diverse and diagnostic in nature. Prefer questions that would be asked by a trained pathologist or for use in a tumor board setting.
"""

SURGERY_SYSTEM_PROMPT = "You are an expert oncologist"
surgery_user_prompt = lambda patient_surgery_report_text, patient_surgery_descriptions_text: f"""
You are provided with the following patient information: 

Patient surgery report {patient_surgery_report_text}
Patient surgery descriptions {patient_surgery_descriptions_text}

Guidelines:
- The patient just underwent a surgery. You MUST start by writing some context using [CONTEXT] tag it.
- The contxt MUST summarize the surgery report and the main outcome of the surgery.
- Do not include greeting text
"""


def read_patient(patient_id):
    """
    Read and print patient data for a given patient ID.
    """
    patient_files = [f for f in os.listdir(os.path.join(cases_path, patient_id)) if f.endswith(".jpg") or f.endswith(".png")]
    patient_clinical_data = json.load(open(os.path.join(cases_path, patient_id, "patient_clinical_data.json")))
    patient_pathological_data = json.load(open(os.path.join(cases_path, patient_id, "patient_pathological_data.json")))
    patient_history_text = json.load(open(os.path.join(cases_path, patient_id, "history_text.txt")))
    patient_icd_codes = json.load(open(os.path.join(cases_path, patient_id, "icd_codes.json")))
    patient_ops_codes = json.load(open(os.path.join(cases_path, patient_id, "ops_codes.json")))
    patient_surgery_report_text = json.load(open(os.path.join(cases_path, patient_id, "surgery_report.txt")))
    patient_surgery_descriptions_text = json.load(open(os.path.join(cases_path, patient_id, "surgery_descriptions.txt")))
    patient_tma_measurements = json.load(open(os.path.join(cases_path, patient_id, "patient_tma_measurements.txt")))
    patient_blood_data = json.load(open(os.path.join(cases_path, patient_id, "patient_blood_data.json")))
    blood_data_reference_ranges = json.load(open(os.path.join(cases_path, patient_id, "blood_data_reference_ranges.json")))
    
    print("Clinical data short")
    short_keys = ["year_of_initial_diagnosis", "age_at_initial_diagnosis", "sex", "smoking_status"]
    patient_clinical_data_short = {key: patient_clinical_data[key] for key in short_keys}
    print(patient_clinical_data_short)
    print("Clinical data")
    print(patient_clinical_data)
    print("Pathological data")
    print(patient_pathological_data)
    print("History text")
    print(patient_history_text)
    print("ICD codes")
    print(patient_icd_codes)
    print("Ops codes")
    print(patient_ops_codes)
    print("Surgery report text")
    print(patient_surgery_report_text)
    print("Surgery descriptions text")
    print(patient_surgery_descriptions_text)
    print("TMA measurements")
    print(patient_tma_measurements)
    print("Blood data")
    print(patient_blood_data)
    print("Blood data reference ranges")
    print(blood_data_reference_ranges)
    print("Files")
    print(patient_files)


def generate_and_parse_context(cases_path, patient_id):
    """
    Generate and parse context for a given patient ID.
    """
    system_prompt = CONTEXT_SYSTEM_PROMPT
    patient_clinical_data = json.load(open(os.path.join(cases_path, patient_id, "patient_clinical_data.json")))
    short_keys = ["year_of_initial_diagnosis", "age_at_initial_diagnosis", "sex", "smoking_status"]
    patient_clinical_data_short = {key: patient_clinical_data[key] for key in short_keys}
    patient_history_text = json.load(open(os.path.join(cases_path, patient_id, "history_text.txt")))
    user_prompt = context_user_prompt(patient_clinical_data_short, patient_history_text)

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7
    )
    output = response.choices[0].message.content

    return output.replace("[CONTEXT]", "").strip()


def generate_and_parse_he_ihc_questions(cases_path, patient_id):
    """
    Generate and parse HE and IHC questions for a given patient ID.
    """
    system_prompt = HE_IHC_SYSTEM_PROMPT
    patient_tma_measurements = json.load(open(os.path.join(cases_path, patient_id, "patient_tma_measurements.txt")))
    patient_pathological_data = json.load(open(os.path.join(cases_path, patient_id, "patient_pathological_data.json")))
    patient_icd_codes = json.load(open(os.path.join(cases_path, patient_id, "icd_codes.json")))
    user_prompt = he_ihc_user_prompt(patient_tma_measurements, patient_pathological_data, patient_icd_codes)

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7
    )
    output = response.choices[0].message.content

    # Extract questions and answers using regex
    question_answer_pairs = re.findall(
        r"\[QUESTION(?: (\d+))?\](.*?)\[ANSWER(?: \1)?\](.*?)(?=\[QUESTION(?: (\d+))?\]|\Z)",
        output,
        re.DOTALL,
    )
    file_paths = []
    for file_path in os.listdir(os.path.join(cases_path, patient_id)):
        if file_path.endswith(".jpg") or file_path.endswith(".png"):
            file_paths.append(os.path.join(cases_path, patient_id, file_path))

    return file_paths, [
        {"question": q.strip(), "answer": a.strip()}
        for _, q, a, _ in question_answer_pairs
    ]


def generate_and_parse_blood_questions(cases_path, patient_id):
    """
    Generate and parse blood questions for a given patient ID.
    """
    system_prompt = BLOOD_SYSTEM_PROMPT
    patient_clinical_data = json.load(open(os.path.join(cases_path, patient_id, "patient_clinical_data.json")))
    short_keys = ["year_of_initial_diagnosis", "age_at_initial_diagnosis", "sex", "smoking_status"]
    patient_clinical_data_short = {key: patient_clinical_data[key] for key in short_keys}
    patient_pathological_data = json.load(open(os.path.join(cases_path, patient_id, "patient_pathological_data.json")))
    patient_blood_data_for_question_generation = json.load(open(os.path.join(cases_path, patient_id, "patient_blood_data.json")))
    blood_data_reference_ranges = json.load(open(os.path.join(cases_path, patient_id, "blood_data_reference_ranges.json")))
    user_prompt = blood_user_prompt(patient_clinical_data_short, patient_pathological_data, patient_blood_data_for_question_generation, blood_data_reference_ranges)
   
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7
    )
    output = response.choices[0].message.content

    # Extract questions and answers using regex
    context_match = re.search(
        r"\[CONTEXT\](.*?)(?=\n\[QUESTION(?: \d+)?\])",
        output,
        re.DOTALL,
    )
    context = context_match.group(1).strip() if context_match else ""
    question_answer_pairs = re.findall(
        r"\[QUESTION(?: (\d+))?\](.*?)\[ANSWER(?: \1)?\](.*?)(?=\[QUESTION(?: (\d+))?\]|\Z)",
        output,
        re.DOTALL,
    )

    file_paths = []
    for file_path in os.listdir(os.path.join(cases_path, patient_id)):
        if file_path.endswith(".jpg") or file_path.endswith(".png"):
            file_paths.append(os.path.join(cases_path, patient_id, file_path))
    file_paths.append(os.path.join(cases_path, patient_id, "patient_blood_data.json"))
    file_paths.append(os.path.join(cases_path, patient_id, "blood_data_reference_ranges.json"))
    file_paths.append(os.path.join(cases_path, patient_id, "patient_pathological_data.json"))
    file_paths.append(os.path.join(cases_path, patient_id, "icd_codes.json"))

    return context, file_paths, [
        {"question": q.strip(), "answer": a.strip()}
        for _, q, a, _ in question_answer_pairs
    ]


def generate_and_parse_surgery_questions(cases_path, patient_id):
    """
    Generate and parse surgery questions for a given patient ID.
    """
    system_prompt = SURGERY_SYSTEM_PROMPT
    patient_surgery_report_text = json.load(open(os.path.join(cases_path, patient_id, "surgery_report.txt")))
    patient_surgery_descriptions_text = json.load(open(os.path.join(cases_path, patient_id, "surgery_descriptions.txt")))
    user_prompt = surgery_user_prompt(patient_surgery_report_text, patient_surgery_descriptions_text)
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7
    )
    output = response.choices[0].message.content
    context = output.replace("[CONTEXT]", "").strip()

    file_paths = []
    for file_path in os.listdir(os.path.join(cases_path, patient_id)):
        if file_path.endswith(".jpg") or file_path.endswith(".png"):
            file_paths.append(os.path.join(cases_path, patient_id, file_path))
    file_paths.append(os.path.join(cases_path, patient_id, "surgery_report.txt"))
    file_paths.append(os.path.join(cases_path, patient_id, "surgery_descriptions.txt"))
    file_paths.append(os.path.join(cases_path, patient_id, "ops_codes.json"))
    file_paths.append(os.path.join(cases_path, patient_id, "patient_pathological_data.json"))
    file_paths.append(os.path.join(cases_path, patient_id, "patient_clinical_data.json"))
    file_paths.append(os.path.join(cases_path, patient_id, "history_text.txt"))
    file_paths.append(os.path.join(cases_path, patient_id, "icd_codes.json"))
    file_paths.append(os.path.join(cases_path, patient_id, "patient_blood_data.json"))
    file_paths.append(os.path.join(cases_path, patient_id, "blood_data_reference_ranges.json"))

    return context, file_paths


def generate_and_parse_questions(cases_path, patient_id):
    """
    Generate and parse questions for a given patient ID.
    """

    # Generate initial patient context
    patient_context = generate_and_parse_context(cases_path, patient_id)
    print("======================")
    print(patient_context)

    # Generate HE and IHC questions
    he_ihc_file_paths, he_ihc_questions = generate_and_parse_he_ihc_questions(cases_path, patient_id)
    print("======================")
    print(he_ihc_questions)
    print(he_ihc_file_paths)

    # Generate blood questions
    blood_context, blood_file_paths, blood_questions = generate_and_parse_blood_questions(cases_path, patient_id)
    print("======================")
    print(blood_context)
    print(blood_file_paths)
    print(blood_questions)

    # Generate surgery questions
    surgery_context, surgery_file_paths = generate_and_parse_surgery_questions(cases_path, patient_id)
    print("======================")
    print(surgery_context)
    print(surgery_file_paths)

    # Combine all questions and file paths
    patient_case_json = []
    patient_case_json.append({"context": patient_context})
    patient_case_json.append({"file_paths": he_ihc_file_paths})
    patient_case_json.extend(he_ihc_questions)
    patient_case_json.append({"context": blood_context})
    patient_case_json.append({"file_paths": blood_file_paths})
    patient_case_json.extend(blood_questions)
    patient_case_json.append({"context": surgery_context})
    patient_case_json.append({"file_paths": surgery_file_paths})

    # Survival and recurrence questions
    hancock_patient = HancockPatient(patient_id)
    patient_clinical_data = hancock_patient.to_json()["clinical_data"]
    SURVIVAL_QUESTION = "Based on all the available patient information, do you think the patient will be alive in 5 years?\nA) Yes\nB) No"
    RECURRENCE_QUESTION = "Based on all the available patient information, do you think the patient's cancer will have a recurrence in 2 years?\nA) Yes\nB) No"
    survival = calculate_survival(patient_clinical_data, 5)
    SURVIVAL_ANSWER = "A) Yes" if survival else "B) No"
    recurrence = calculate_recurrence(patient_clinical_data, 2)
    RECURRENCE_ANSWER = "A) Yes" if recurrence else "B) No"
    patient_case_json.append({
        "question": SURVIVAL_QUESTION,
        "answer": SURVIVAL_ANSWER
    })
    patient_case_json.append({
        "question": RECURRENCE_QUESTION,
        "answer": RECURRENCE_ANSWER
    })

    return patient_case_json


def calculate_survival(patient_clinical_data, years):
    """
    Calculate survival based on clinical data and years.
    """
    days_in_year = 365.25
    last_date_alive = days_in_year * years
    if patient_clinical_data["survival_status"] == "deceased" and patient_clinical_data["days_to_last_information"] < last_date_alive:
        return False
    else:
        return True
    

def calculate_recurrence(patient_clinical_data, years):
    """
    Calculate recurrence based on clinical data and years.
    """
    days_in_year = 365.25
    last_date_alive = days_in_year * years
    if patient_clinical_data["recurrence"] == "yes" and patient_clinical_data["days_to_recurrence"] < last_date_alive:
        return True
    else:
        return False
    

if __name__ == "__main__":
    conf = OmegaConf.load("neurips25/configs/base.yaml")
    archive_path = conf.hancock.archive_path
    extract_path = conf.hancock.extract_path
    thumbnails_path = conf.hancock.thumbnails_path
    cases_path = conf.hancock.cases_path

    patient_id = "346"
    # Read patient data only for printing purposes
    read_patient(patient_id)

    # Setup OpenAI client
    openai_token = conf.openai_token
    client = openai.OpenAI(api_key=openai_token)

    # Generate patient questions
    questions = {}
    for patient_id in [296, 741, 476, 162, 583, 564, 334, 176, 121, 698, 403, 761, 706, 740, 664, 346, 530, 120, 342, 225, 116, 559, 632, 104, 723, 250, 606]:
        patient_id = str(patient_id)
        questions[patient_id] = generate_and_parse_questions(cases_path, patient_id)

    # Save questions to JSON file
    with open(conf.dataset.path, "w") as f:
        json.dump(questions, f)
        f.write("\n")
    