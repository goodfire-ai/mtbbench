import json
import gc
import os
import torch
from omegaconf import OmegaConf

from neurips25.models.agent import DoctorAgent
from neurips25.models.agent_with_tools import DoctorAgentWithTools
from neurips25.models.agent_with_tools_msk import DoctorAgentWithToolsMsk
from neurips25.utils.args import get_parser
from neurips25.utils.load_model import get_model


if __name__ == "__main__":
    args = get_parser()
    os.makedirs(args.output_dir, exist_ok=True)
    conf = OmegaConf.load("neurips25/configs/base.yaml")
    # Load the patient cases from the JSON file
    if args.dataset == "hancock":
        patient_cases = json.loads(open(conf.dataset.path, "r").readlines()[0])
    elif args.dataset == "msk":
        patient_cases = json.loads(open(conf.dataset.msk_bench, "r").readlines()[0])

    # Count the number of questions in all the patient cases
    question_count = 0
    for patient_case in patient_cases:
        for item in patient_cases[patient_case]:
            if "question" in item:
                question_count += 1
    print(question_count)

    # Initialize the main LLM and oracle LLM
    main_llm, model_name = get_model(args.doctor_model)

    # Loop through the patient cases and run the agent
    processed = 0
    for case_id in list(patient_cases.keys()):
        # Stop early once we've processed the requested number of newly-run cases
        # (used by the smoke / reproduction harness; default None runs the full set).
        if args.max_cases is not None and processed >= args.max_cases:
            print(f"Reached --max-cases={args.max_cases}, stopping.")
            break

        logs = [x.split("_")[0] for x in os.listdir(args.output_dir)]
        if case_id in logs:
            print(f"Case {case_id} already exists, skipping...")
            continue

        # How to see how much memory allocated to torch model GPU RAM
        print(torch.cuda.memory_allocated() / 1024 ** 3, "GB")

        # Select the agent. By default the base, no-tools DoctorAgent is used; passing
        # --use-tools selects the tool-augmented agent for the dataset (HANCOCK: CONCH +
        # IHC density tool; MSK: the MSK tool set).
        if args.use_tools:
            if args.dataset == "hancock":
                agent = DoctorAgentWithTools(main_llm, main_llm, model_name=model_name, output_dir=args.output_dir)
            elif args.dataset == "msk":
                agent = DoctorAgentWithToolsMsk(main_llm, main_llm, model_name=model_name, output_dir=args.output_dir)
            else:
                raise ValueError(f"--use-tools given but no tool agent is defined for dataset {args.dataset!r}")
        else:
            agent = DoctorAgent(main_llm, main_llm, model_name=model_name, output_dir=args.output_dir)
        case_data = patient_cases[case_id]
        chat = agent.run_case(case_data=case_data, case_id=case_id)
        processed += 1

        # How to see how much memory allocated to torch model GPU RAM
        print(torch.cuda.memory_allocated() / 1024 ** 3, "GB")

        # Empty the cache
        del agent
        gc.collect()
        torch.cuda.empty_cache()