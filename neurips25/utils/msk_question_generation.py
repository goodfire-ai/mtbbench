from neurips25.utils.patient import Patient
import json
import os
from omegaconf import OmegaConf
import openai

conf = OmegaConf.load("neurips25/configs/base.yaml")
cases_path = conf.msk.cases_path
questions_path = conf.dataset.msk_bench
openai_token = conf.openai_token
client = openai.OpenAI(api_key=openai_token)

interesting_patients = ["P-0004727", "P-0024579", "P-0005708", "P-0006191", "P-0006687",
                        "P-0006691", "P-0007637", "P-0005014", "P-0009786", "P-0022292",
                        "P-0022653", "P-0022826", "P-0022916", "P-0024579", "P-0027883",
                        "P-0028198", "P-0031600", "P-0032535", "P-0035737", "P-0039020"]

datapoints = {
    "P-0004727": [((60, 61.7), [{"recurrence": 24, "answer": "yes"}]), ((61.7, 63), [{"alive": 12, "answer": "no"}])],
    "P-0024579": [((71, 73.1), [{"recurrence": 24, "answer": "yes"}]), ((73.1, 74.1), [{"alive": 12, "answer": "no"}])],
    "P-0005708": [((58, 58.6), [{"recurrence": 12, "answer": "yes"}, {"alive": 12, "answer": "no"}])],
    "P-0006191": [((55, 56.1), [{"recurrence": 12, "answer": "yes"}, {"alive": 24, "answer": "yes"}]), ((56.1, 59.8), [{"alive": 12, "answer": "yes"}, {"progress": 12, "answer": "no"}])],
    "P-0006687": [((53, 54.5), [{"recurrence": 12, "answer": "yes"}]), ((54.5, 56), [{"recurrence": 12, "answer": "no"}, {"recurrence": 24, "answer": "yes"}]), ((56, 59.1), [{"alive": 12, "answer": "no"}])],
    "P-0006691": [((63, 64), [{"progress": 6, "answer": "no"}, {"recurrence": 6, "answer": "no"}]), ((64, 65.5), [{"progress": 6, "answer": "no"}, {"alive": 12, "answer": "yes"}]), ((65.5, 66.6), [{"alive": 12, "answer": "no"}])],
    "P-0007637": [((53, 55.8), [{"recurrence": 18, "answer": "no"}]), ((55.8, 58.1), [{"alive": 12, "answer": "yes"}, {"progress": 6, "answer": "yes"}]), ((58.1, 59), [{"alive": 9, "answer": "no"}])],
    "P-0005014": [((75.5, 80.1), [{"recurrence": 24, "answer": "no"}, {"progress": 24, "answer": "no"}, {"alive": 36, "answer": "yes"}]), ((80.1, 85), [{"alive": 24, "answer": "yes"}, {"progress": 12, "answer": "yes"}])],
    "P-0009786": [((51.9, 53), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}]), ((53, 57), [{"alive": 12, "answer": "yes"}, {"recurrence": 12, "answer": "yes"}]), ((57, 58), [{"alive": 24, "answer": "yes"}])],
    "P-0022292": [((65.5, 66.4), [{"progress": 12, "answer": "no"}, {"recurrence": 12, "answer": "yes"}]), ((66.4, 67.8), [{"progress": 12, "answer": "yes"}, {"recurrence": 12, "answer": "yes"}, {"alive": 12, "answer":"yes"}]), ((67.8, 68.9), [{"alive": 6, "answer": "no"}])],
    "P-0022653": [((40.6, 47.1), [{"progress": 24, "answer": "no"}, {"alive": 24, "answer": "yes"}]), ((47.1, 50), [{"progress": 12, "answer": "no"}, {"alive": 24, "answer": "yes"}]), ((50, 52), [{"alive": 12, "answer": "yes"}])],
    "P-0022826": [((60.8, 61), [{"recurrence": 6, "answer": "no"}, {"alive": 6, "answer": "yes"}]), ((61, 62), [{"alive": 12, "answer": "no"}, {"recurrence": 12, "answer": "yes"}, {"progress": 12, "answer": "yes"}])],
    "P-0022916": [((54.8, 56), [{"recurrence": 9, "answer": "yes"}]), ((56, 57.8), [{"alive": 12, "answer": "yes", "progress": 12, "answer": "yes"}]), ((57.8, 59), [{"alive": 12, "answer": "yes", "progress": 12, "answer": "no", "recurrence": 12, "answer": "no"}]), ((59, 60), [{"alive": 12, "answer": "yes"}])],
    "P-0024579": [((71.7, 72.2), [{"progression": 12, "answer": "no"}, {"alive": 12, "answer": "yes"}]), ((72.2, 72.9), [{"alive": 3, "answer": "yes"}, {"alive": 6, "answer": "no"}])],
    "P-0027883": [((48.5, 49.1), [{"recurrence": 6, "answer": "yes"}, {"progress": 6, "answer": "yes"}]), ((49.1, 50.2), [{"alive": 12, "answer": "yes"}, {"progress": 12, "answer": "no"}]), ((50.2, 51.3), [{"alive": 12, "answer": "yes"}, {"progress": 12, "answer": "no"}, {"recurrence": 12, "answer": "no"}]), ((51.3, 53), [{"alive": 12, "answer": "yes"}, {"progress": 12, "answer": "yes"}])],
    "P-0028198": [((73.5, 74.7), [{"recurrence": 12, "answer": "no"}]), ((74.7, 75.7), [{"recurrence": 12, "answer": "yes"}]), ((75.7, 77), [{"alive": 12, "answer": "yes"}, {"progess": 12, "answer": "no"}])],
    "P-0031600": [((67.7, 71.6), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}, {"alive": 18, "answer": "yes"}])],
    "P-0032535": [((61.5, 62.3), [{"recurrence": 6, "answer": "no"}, {"progress": 6, "answer": "no"}]), ((62.3, 63.2), [{"recurrence": 6, "answer": "yes"}, {"progress": 6, "answer": "yes"}, {"alive": 12, "answer": "yes"}]), ((63.2, 64.6), [{"alive": 12, "answer": "no"}])],
    "P-0035737": [((70, 71.2), [{"recurrence": 6, "answer": "yes"}, {"progress": 12, "answer": "yes"}, {"alive": 12, "answer": "yes"}])],
    "P-0039020": [((62.2, 63.1), [{"recurrence": 6, "answer": "yes"}, {"progress": 6, "answer": "yes"}]), ((63.1, 63.8), [{"alive": 6, "answer": "no"}])],
    "P-0039147": [((71, 72.7), [{"recurrence": 6, "answer": "yes"}, {"progress": 6, "answer": "yes"}]), ((72.7, 73.9), [{"progress": 6, "answer": "yes"}, {"alive": 6, "answer": "yes"}])],
    "P-0040042": [((60.3, 61.5), [{"recurrence": 12, "answer": "yes"}, {"progress": 12, "answer": "yes"}]), ((61.5, 64.2), [{"alive": 12, "answer": "yes"}]), ((64.2, 65.2), [{"alive": 12, "answer": "yes"}]), ((65.2, 65.8), [{"alive": 12, "answer": "no"}])],
    "P-0040055": [((62.3, 63.3), [{"recurrence": 12, "answer": "yes"}, {"progress": 12, "answer": "yes"}]), ((63.3, 63.7), [{"alive": 18, "answer": "yes"}])],
    "P-0040144": [((67.9, 69.5), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}, {"alive": 18, "answer": "yes"}])],
    "P-0040302": [((50.8, 51.6), [{"recurrence": 6, "answer": "yes"}, {"progress": 6, "answer": "yes"}]), ((51.6, 52.4), [{"progress": 4, "answer": "yes"}, {"alive": 12, "answer": "no"}])],
    "P-0040923": [((69, 70), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}]), ((70, 72), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}, {"alive": 12, "answer": "yes"}])],
    "P-0040957": [((69.5, 70.5), [{"recurrence": 12, "answer": "yes"}, {"progress": 12, "answer": "yes"}, {"alive": 18, "answer": "yes"}]), ((70.5, 72.5), [{"alive": 12, "answer": "yes"}])],
    "P-0041210": [((78.7, 79.3), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}]), ((79.3, 80.5), [{"recurrence": 9, "answer": "no"}, {"progress": 9, "answer": "no"}]), ((80.5, 81.5), [{"alive": 12, "answer": "yes"}]), ((81.5, 81.7), [{"alive": 12, "answer": "no"}])],
    "P-0041381": [((50.4, 52), [{"recurrence": 12, "answer": "yes"}, {"alive": 12, "answer": "yes"}]), ((52, 54), [{"progress": 12, "answer": "no"}, {"alive": 12, "answer": "yes"}]), ((54, 55), [{"alive": 12, "answer": "no"}]), ((55, 56), [{"alive": 12, "answer": "yes"}]), ((56, 57.4), [{"progress": 12, "answer": "yes"}, {"recurrence": 12, "answer": "yes"}, {"alive": 12, "answer": "yes"}]), ((57.4, 59), [{"alive": 15, "answer": "yes"}]), ((59, 60.7), [{"alive": 9, "answer": "no"}])],
    "P-0041506": [((65.3, 66.6), [{"recurrence": 6, "answer": "yes"}]), ((66.6, 68), [{"progress": 12, "answer": "yes"}, {"alive": 12, "answer": "no"}])],
    "P-0042090": [((50.8, 51.5), [{"progress": 12, "answer": "no"}, {"alive": 12, "answer": "yes"}]), ((51.8, 53.1), [{"recurrence": 12, "answer": "yes"}, {"alive": 12, "answer": "yes"}])],
    "P-0042530": [((60.5, 61.6), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}, {"alive": 12, "answer": "yes"}]), ((61.6, 62.6), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}, {"alive": 12, "answer": "yes"}])],
    "P-0042652": [((75.3, 77.4), [{"recurrence": 12, "answer": "yes"}, {"progress": 12, "answer": "yes"}]), ((77.4, 78.8), [{"alive": 12, "answer": "no"}])],
    "P-0043363": [((60, 61.3), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}]), ((61.3, 63), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}, {"alive": 12, "answer": "no"}])],
    "P-0043580": [((66.4, 67.3), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}]), ((67.3, 69), [{"recurrence": 6, "answer": "yes"}, {"progress": 6, "answer": "yes"}, {"alive": 6, "answer": "yes"}]), ((69, 69.9), [{"progress": 6, "answer": "yes"}, {"alive": 6, "answer": "no"}])],
    "P-0043590": [((59.6, 61), [{"recurrence": 6, "answer": "yes"}, {"progress": 6, "answer": "no"}])],
    "P-0044015": [((47.6, 48.3), [{"recurrence": 6, "answer": "yes"}, {"progress": 6, "answer": "no"}]), ((48.3, 50), [{"alive": 12, "answer": "yes"}, {"progress": 12, "answer": "no"}]), ((50, 50.8), [{"alive": 12, "answer": "yes"}, {"progress": 12, "answer": "no"}, {"recurrence": 12, "answer": "no"}])],
    "P-0044063": [((44.4, 45.5), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}, {"alive": 12, "answer": "yes"}])],
    "P-0044328": [((49.7, 52), [{"recurrence": 12, "answer": "no"}, {"progress": 12, "answer": "no"}, {"alive": 12, "answer": "yes"}]), ((52, 53), [{"recurrence": 6, "answer": "no"}, {"progress": 6, "answer": "no"}, {"alive": 9, "answer": "yes"}])],
    "P-0044449": [((58, 61.4), [{"progress": 3, "answer": "yes"}, {"alive": 12, "answer": "yes"}]), ((61.4, 64.1), [{"recurrence": 24, "answer": "no"}, {"progress": 24, "answer": "no"}, {"alive": 24, "answer": "yes"}])],
    "P-0044903": [((66, 67.3), [{"recurrence": 24, "answer": "no"}, {"progress": 24, "answer": "no"}, {"alive": 24, "answer": "yes"}])],
}

MSK_CONTEXT_SYSTEM_PROMPT = """
You are a medical assistant that is expert at combining information provided to you by the user.
"""
MSK_CONTEXT_USER_PROMPT = lambda patient_clinical, patient_sample, additional: f"""
You are given the following patient information:
Paient clinical information: {patient_clinical}
Patient sample information: {patient_sample}
Additional information to include: {additional}

You must combine this information in a single paragraph.
The summary must sound concise, natural, and as if it were written by a medical doctor.
"""

MSK_FOLLOWING_CONTEXT_USER_PROMPT = lambda additional: f"""
You are given the following information:
{additional}
You must combine this information in a single paragraph.
The summary must sound concise, natural, and as if it were written by a medical doctor.
"""

def generate_initial_patient_context(patient_clinical, patient_sample, additional):
    # Mention files will be available to review
    messages=[
        {"role": "system", "content": MSK_CONTEXT_SYSTEM_PROMPT},
        {"role": "user", "content": MSK_CONTEXT_USER_PROMPT(patient_clinical, patient_sample, additional)},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
        temperature=0.7
    )
    output = response.choices[0].message.content
    return output


def generate_following_context(context):
    
    messages=[
        {"role": "system", "content": MSK_CONTEXT_SYSTEM_PROMPT},
        {"role": "user", "content": MSK_FOLLOWING_CONTEXT_USER_PROMPT(context)},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
        temperature=0.7
    )
    output = response.choices[0].message.content
    return output


def generate_recurrence_question(months, answer, age):
    question = f"Given that the patient is {age} years old, will the cancer have a recurrence in the next {months} months?\nA) Yes\nB) No"
    if answer == "yes":
        answer = "A) Yes"
    else:
        answer = "B) No"
    return {"question": question, "answer": answer}


def generate_alive_question(months, answer, age):
    question = f"Given that the patient is {age} years old, will the patient be still alive in the next {months} months?\nA) Yes\nB) No"
    if answer == "yes":
        answer = "A) Yes"
    else:
        answer = "B) No"
    return {"question": question, "answer": answer}


def generate_progress_question(months, answer, age):
    question = f"Given that the patient is {age} years old, will the cancer progress in the next {months} months?\nA) Yes\nB) No"
    if answer == "yes":
        answer = "A) Yes"
    else:
        answer = "B) No"
    return {"question": question, "answer": answer}


dataset = {}

for key in datapoints:
    print(key)
    # Make cases directory if it doesn't exist
    if not os.path.exists(os.path.join(cases_path, key)):
        os.makedirs(os.path.join(cases_path, key))

    dataset[key] = []
    patient = Patient(key)
    patient_json = patient.to_json()

    # Initial patient information for context
    patient_clinical = patient.clinical_patient.to_dict(orient="records")[0]
    patient_clinical_keys = ["GENDER", "RACE", "ETHNICITY"]
    patient_clinical = {k: patient_clinical[k] for k in patient_clinical_keys}
    patient_clinical["AGE"] = datapoints[key][0][0][0]
    patient_sample = patient.samples[0].clinical_sample.to_dict(orient="records")[0]
    patient_sample_keys = ["CANCER_TYPE_DETAILED"]
    patient_sample = {k: patient_sample[k] for k in patient_sample_keys}
    # File paths for the timeline and other data
    file_paths = set()
    context = ""
    for idx, ((start, end), questions) in enumerate(datapoints[key]):
        # Extract timeline events
        timeline = patient.summarize_events(patient.get_sorted_events(start, end))
        file_paths.add(os.path.join(cases_path, key, f"timeline{idx}.txt"))
        with open(os.path.join(cases_path, key, f"timeline{idx}.txt"), "w") as f:
            f.write(timeline)

        # If sample is sequenced, save gene and sample data
        if "specimen -->" in timeline:
            context += "Mention that the patient has a sample sequenced.\n"
            patient_specimen_sequenced = patient.samples[0].clinical_sample.to_dict(orient="records")[0]
            sample_cna = patient.samples[0].cna
            sample_mutation = patient.samples[0].mutation
            sample_sv = patient.samples[0].sv

            file_paths.add(os.path.join(cases_path, key, "specimen.txt"))
            with open(os.path.join(cases_path, key, "specimen.txt"), "w") as f:
                json.dump(patient_specimen_sequenced, f)
            if len(sample_cna) > 0:
                context += "Mention that the patient has a CNA data available.\n"
                file_paths.add(os.path.join(cases_path, key, "cna.csv"))
                sample_cna.to_csv(os.path.join(cases_path, key, "cna.csv"), index=False)
            if len(sample_mutation) > 0:
                context += "Mention that the patient has a mutation data available.\n"
                file_paths.add(os.path.join(cases_path, key, "mutation.csv"))
                sample_mutation.to_csv(os.path.join(cases_path, key, "mutation.csv"), index=False)
            if len(sample_sv) > 0: 
                context += "Mention that the patient has a SV data available.\n"
                file_paths.add(os.path.join(cases_path, key, "sv.csv"))
                sample_sv.to_csv(os.path.join(cases_path, key, "sv.csv"), index=False)

        context += f"Mention patient history is available about diagnosis, treatments, lab tests for the period from {start} to {end} years old.\n"
        if idx == 0:
            context = generate_initial_patient_context(patient_clinical, patient_sample, context)
        else:
            context = generate_following_context(context)
        dataset[key].append({"context": context})
        dataset[key].append({"file_paths": list(file_paths)})
        # Generate questions based on the datapoints
        for question in questions:
            if "recurrence" in question:
                dataset[key].append(generate_recurrence_question(question["recurrence"], question["answer"], end))
            if "alive" in question:
                dataset[key].append(generate_alive_question(question["alive"], question["answer"], end))
            if "progress" in question:
                dataset[key].append(generate_progress_question(question["progress"], question["answer"], end))

        if idx < len(datapoints[key]) - 1:
            context = ""
            for question in questions:
                if "recurrence" in question.keys():
                    context += f"Mention cancer has recurred in {question['recurrence']} months: {question['answer']}\n"
                if "progress" in question.keys():
                    context += f"Mention cancer has progressed in {question['progress']} months: {question['answer']}\n"
print(dataset)