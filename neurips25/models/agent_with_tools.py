import json
import os
import re
import time
import torch
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from pathlib import Path
from neurips25.eval import *
from neurips25.tools.conch import Conch

class DoctorAgentWithTools:
    def __init__(self, main_llm, oracle_llm, model_name, output_dir="./agent_logs"):
        """
        Initialize the PatientAgent with the main LLM and oracle LLM for answers.
        """
        self.main_llm = main_llm
        self.oracle_llm = oracle_llm
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chat_history = []
        self.strike_count = 0
        self.conch = Conch(device="cuda")


    def _use_conch(self, image_name, options, current_file_paths):
        """
        Use the CONCH model to analyze the image and options.
        """
        image_name = image_name.replace(" ", "")
        image_path = next((fp for fp in current_file_paths if os.path.basename(fp) == image_name), None)
        logger.info(f"Using CONCH model for image {image_name} with options {options}")
        if not image_path:
            logger.info(f"Image {image_name} not found in the available files.")
            return f"[ERROR: Image {image_name} not found in the available files.]"
        
        image_path = image_path.replace(".jpg", "_CONCH.jpg")
        predicted_class, sim_scores = self.conch.image_to_text_retrieval(
            image_path=image_path,
            prompts=options
        )

        probability_label = ""
        probability = round(max(sim_scores[0]), 2)
        if 0.0 <= probability < 0.2:
            probability_label = "very low"
        elif 0.2 <= probability < 0.4:
            probability_label = "low"
        elif 0.4 <= probability < 0.6:
            probability_label = "medium"
        elif 0.6 <= probability < 0.8:
            probability_label = "high"
        elif 0.8 <= probability <= 1.0:
            probability_label = "very high"
        reply = f"The image resembles {predicted_class} with {probability_label} probability."        
        return reply


    def _use_ihctool(self, image_name, current_file_paths):
        """
        Use the IHC model to analyze the image.
        """
        image_name = image_name.replace(" ", "")
        image_path = next((fp for fp in current_file_paths if os.path.basename(fp) == image_name), None)
        logger.info(f"Using IHCTool model for image {image_name}")
        if not image_path:
            logger.info(f"Image {image_name} not found in the available files.")
            return f"[ERROR: Image {image_name} not found in the available files.]"
        
        image_path = self.case_id + "_" + image_name
        cell_measurements = pd.read_csv("data/hancock/cell_density_measurements.csv")
        cell_measurements = cell_measurements[cell_measurements["file_name"] == image_path]["value"].values
        if len(cell_measurements) == 0:
            logger.info(f"Image {image_name} not found in the IHCTool data.")
            return f"Image {image_name} cannot be analyzed by IHCTool. You will have to do it manually."
        cell_measurements = cell_measurements[0]
        return f"According to IHC tool around {cell_measurements}% of the cells in {image_name} are positively stained."


    def _parse_files(self, file_paths):
        """
        Parse the file paths to separate images and text files.
        """
        images = []
        texts = []
        
        for path in file_paths:
            ext = Path(path).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                images.append(path)
            elif ext in ['.txt', '.json']:
                texts.append(path)

        file_overview = []
        for img in images:
            file_overview.append(f"[FILE: {os.path.basename(img)}]")
        for txt_path in texts:
            file_overview.append(f"[FILE: {os.path.basename(txt_path)}]")

        return file_overview
        

    def _attach_files(self, requested_files, current_file_basenames, current_file_paths):
        """
        Attach files to the conversation based on the requested files.
        """
        file_msg = ""
        attached_files = []
        hallucinated_files = []
        all_given_files = []
        ihc_flag = False

        for req_file in set(requested_files):
            req_file = req_file.replace(" ", "") # some models add spaces before extension
            # Check if the requested file is in the current file basenames
            if req_file not in current_file_basenames:
                file_msg += f"[FILE: {req_file}] not found. Only ask for files that were listed to you earlier! Example request format for 2 images: [REQUEST: image1.jpg] [REQUEST: image2.jpg]\n"
                hallucinated_files.append(req_file)
                continue

            # Add to all given files
            all_given_files.append(req_file)
            
            # Check if the requested file is in the current file paths
            full_path = next((fp for fp in current_file_paths if os.path.basename(fp) == req_file), None)
            if full_path:
                # Add H&E path
                if full_path.lower().endswith(('.jpg', '.jpeg')):
                    attached_files.append(full_path)
                    file_msg += f"[FILE: {req_file}] included in your context\n"
                # Add IHC path
                elif full_path.lower().endswith(('.png')):
                    ihc_flag = True
                    attached_files.append(full_path)
                    file_msg += f"[FILE: {req_file}] included in your context\n"
                    ihc_msg = self._use_ihctool(req_file, current_file_paths)
                    file_msg += f"[IHCTool: {ihc_msg}]\n"
                # Else text file, so read it
                else:
                    try:
                        with open(full_path, 'r') as f:
                            content = json.load(f) if full_path.endswith('.json') else f.read()
                            file_msg += f"[FILE: {req_file}] included in your context\n{content}\n"
                    except Exception as e:
                        file_msg += f"[ERROR: Failed to read {req_file}: {e}]\n"

        if ihc_flag:
            file_msg += "[IHCTool: You can use this information, but keep in mind that IHCTool is not perfect and you MUST analyze the images yourself too.]\n"
        return file_msg, attached_files, hallucinated_files, all_given_files
    

    def _dettach_files(self, conversation):
        """
        Detach files from the conversation.
        """
        for entry in conversation:
            if 'files' in entry:
                del entry['files']
            if entry['role'] == 'user':
                content = entry['content']
                file_names = re.findall(r"\[FILE: (.+?)\] included\n", content)
                content = ""
                for file_name in file_names:
                    content += f"[FILE: {file_name}] was accessed by you\n"
                if file_names:
                    entry['content'] = content + "You can access all these files once again if necessary by asking for them in the format [REQUEST: filename.extension].\n"
                    logger.info(f"Files detached from conversation: {file_names}")
                    file_names = []

                # Remove IHC analysis
                ihc_analysis = re.findall(r"\[IHCTool: (.+?)\]", content)
                if ihc_analysis:
                    for ihc in ihc_analysis:
                        content = content.replace(f"[IHCTool: {ihc}]", "")
                    entry['content'] = content
                    ihc_analysis = []
        return conversation
    
    @torch.no_grad()
    def run_case(self, case_data, case_id="patient_case"):
        """
        Run the agent on a single patient case.
        """
        logger.info(f"Running agent on {case_id}")
        self.case_id = case_id

        conversation = []
        self.chat_history = []  # Clear previous runs
        self.strike_count = 0
        # self.question_count = 0

        # System prompt
        system_message = (
            f"You are a pathologist AI assistant expert at analyzing patient data and answering user questions.\n"
            f"You will be provided with files that you are allowed to read.\n"
            f"To ask for files, include in your reasoning [REQUEST: filename.extension] for each file you need"
            f"Example: [REQUEST: image1.jpg] [REQUEST: image2.jpg]\n"
            f"To provide a final answer to a question, include [ANSWER: LETTER) your answer] in your response, specifying the answer choice you picked (A, B, C, D, E, or F).\n"
            f"You MUST ONLY provide [ANSWER] when you have all necessary information."
            f"You also have access to a H&E foundation model CONCH that can be used to determine cancer type and NOTHING ELSE.\n"
            f"To use CONCH you must provide the H&E image name and extension and a list of options in the format [CONCH: filename.extension, (option1 text), (option2 text), ...] with each option surrounded by ()\n"
            f"Example: [CONCH: image1.jpg, (melanoma), (squamous cell carcinoma)] [CONCH: image2.jpg, (option 1), (option 2)]\n"
            f"The model will then tell you which option resembles the image the most.\n"
        )
        conversation.append({"role": "system", "content": system_message})

        # Initialize context and file tracking
        context_message = ""
        current_file_paths = []
        current_file_basenames = []

        context_message = ""
        for entry in case_data:
            # Patient context
            if 'context' in entry:
                # Update current context
                context_message = f"You are given the following new patient information: \n{entry['context']}\n"
            
            # Patient files
            elif 'file_paths' in entry:
                # Update available files
                current_file_paths = entry['file_paths']
                current_file_basenames = [os.path.basename(f) for f in current_file_paths]
                file_overview = self._parse_files(current_file_paths)

                # Announce new available files
                context_message +=( f"New files available:\n {'\n'.join(file_overview)}" +
                                    "Remember that you can ask for files by providing the following tag [REQUEST: filename.extension]. You may also ask for multiple files at once if necessary. If you ask for a file, you MUST WAIT to receive it from the user.\n"
                                    "You must provide a separate [REQUEST] tag for each file you need. Example [REQUEST: image1.jpg] [REQUEST: image2.jpg].\n"
                                    "You can also use the CONCH model to determine cancer type or invasion by providing the H&E image name and extension and a list of options. Example: [CONCH: image1.jpg, (melanoma), (squamous cell carcinoma)] [CONCH: image2.jpg, (option 1), (option 2)].\n"
                                    "The model will then tell you which option resembles the image the most. You can ONLY use this model for H&E.\n"
                                )

            # Patient question
            elif 'question' in entry:
                # Now we can ask a question
                question = entry['question']
                expected_answer = entry['answer']
                # self.question_count += 1

                # Detach files from the conversation
                conversation = self._dettach_files(conversation)

                logger.info(f"Processing question: {question.strip()}")
                conversation.append(
                    {
                        "role": "user", 
                        "content": (
                                    f"{context_message}\n Question: {question}\n"
                                    "Remember that you can ask for files by providing the following tag [REQUEST: filename.extension]. You may also ask for multiple files at once if necessary. If you ask for a file, you MUST WAIT to receive it from the user.\n"
                                    "You must provide a separate [REQUEST] tag for each file you need. Example [REQUEST: image1.jpg] [REQUEST: image2.jpg].\n"
                                    "You can also use the CONCH model to determine cancer type or invasion by providing the H&E image name and extension and a list of options. Example: [CONCH: image1.jpg, (melanoma), (squamous cell carcinoma)] [CONCH: image2.jpg, (option 1), (option 2)].\n"
                                    "The model will then tell you which option resembles the image the most. You can ONLY use this model for H&E.\n"
                                )
                    }
                )
                # Empty the context message since we have included it in the conversation
                context_message = ""

                # Start the conversation loop until a valid answer is provided
                question_start_time = time.time()
                prev_requested_files = []
                all_files_accessed_for_question = []
                all_halucinated_files = []
                while True:
                    response = self.main_llm.evaluate(messages=conversation)
                    logger.debug(f"Model response: {response}")
                    conversation.append({"role": "assistant", "content": response})

                    # If the model requests files and they are different from the previous request
                    requested_files = re.findall(r"\[REQUEST:\s*([^\]]+)\]", response)
                    if requested_files and requested_files != prev_requested_files:
                        prev_requested_files = requested_files 
                        # Check if the files are in list format
                        if len(requested_files) == 1 and requested_files[0].startswith("[REQUEST:") and "," in requested_files[0]:
                            requested_files = requested_files[0].split(",")
                        # Add files to the conversation
                        file_msg, attached_files, hallucinated_files, all_given_files = self._attach_files(requested_files, current_file_basenames, current_file_paths)
                        logger.info(f"Files: {file_msg}")
                        all_files_accessed_for_question.extend(all_given_files)
                        all_halucinated_files.extend(hallucinated_files)

                        suffix = ""
                        conversation.append({"role": "user", "content": file_msg + suffix, "files": attached_files})
                    
                    # If the model uses CONCH
                    elif re.findall(r"\[CONCH: (\S+), (.*)\]", response):

                        matches = re.finditer(r"\[CONCH: (\S+), (.*?)\]", response)
                        conch_responses = []
                        for match in matches:
                            image_name = match.group(1)
                            options_string = match.group(2)
                            options = re.findall(r"\((.*?)\)", options_string)
                            reply = self._use_conch(image_name, options, current_file_paths)
                            
                            conch_responses.append(reply)

                        if conch_responses:
                            combined_reply = "\n".join(conch_responses)
                            combined_reply += "You can use this information, but keep in mind that CONCH is not perfect and you should also use your own reasoning."
                            logger.info(f"CONCH response: {combined_reply}")
                            conversation.append({"role": "user", "content": combined_reply})
                        else:
                            logger.info("Model did not provide a valid CONCH request.")
                            conversation.append(
                                {
                                    "role": "user", 
                                    "content": "You must provide the full options and correct H&E file name to CONCH in the format of [CONCH: filename.extension, (option1 text), (option2 text), ...]"
                                }
                            )

                    # If the model uses IHC
                    elif re.findall(r"\[IHCTool: (\S+)\]", response):
                        matches = re.finditer(r"\[IHCTool: (\S+)\]", response)
                        ihc_responses = []
                        for match in matches:
                            image_name = match.group(1)
                            ihc_response = self._use_ihctool(image_name, current_file_paths)
                            ihc_responses.append(ihc_response)

                        if ihc_responses:
                            combined_reply = "\n".join(ihc_responses)
                            combined_reply += "You can use this information, but keep in mind that IHCTool is not perfect and you should also use your own reasoning."
                            logger.info(f"IHC response: {combined_reply}")
                            conversation.append({"role": "user", "content": combined_reply})
                        else:
                            logger.info("Model did not provide any valid IHC requests.")
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": "You must provide the full file name to IHC in the format of [IHCTool: filename.extension]"
                                }
                            )

                    # If the model provides an answer
                    elif re.findall(r"\[ANSWER:([^\]]+)\]", response):
                        if self.strike_count > 2:
                            logger.info("Model failed to provide a valid answer or request after 3 attempts.")
                            self.strike_count = 0
                            question_end_time = time.time()
                            correct = False
                            self.chat_history.append({
                                "question": question,
                                "answer": expected_answer,
                                "response": response,
                                "correct": correct,
                                "files_accessed": all_files_accessed_for_question,
                                "files_hallucinated": all_halucinated_files,
                                "question_time": question_end_time - question_start_time,
                            })
                            break

                        
                        question_end_time = time.time()
                        # Take first answer only
                        response = re.findall(r"\[ANSWER:\s*([^\]]+)\]", response)[-1]
                        if (len(response.strip()) >=2 and response.strip()[0].lower() in 'abcdef' and response.strip()[1] in [")", "]", " "]) or (len(response.strip()) == 1 and response.strip()[0].lower() in 'abcdef'):
                            self.strike_count = 0
                            correct = expected_answer.strip()[0].lower() == response.strip()[0].lower()
                            self.chat_history.append({
                                "question": question,
                                "answer": expected_answer,
                                "response": response,
                                "correct": correct,
                                "files_accessed": all_files_accessed_for_question,
                                "files_hallucinated": all_halucinated_files,
                                "question_time": question_end_time - question_start_time,
                            })
                            break
                        else:
                            self.strike_count += 1
                            logger.info("Model did not provide a valid answer or request.")
                            conversation.append(
                                {
                                    "role": "user", 
                                    "content": "Please provide the final answer in [ANSWER: LETTER) answer] specifying the answer choice letter you picked (A, B, C, D, E, or F) or ask for files with [REQUEST: filename.extension]. Make sure you have both the opening an closing brackets."
                                }
                            )
                        
                    
                    # If the model does neither
                    else:
                        self.strike_count += 1

                        # If the model has failed to follow our template, break the loop
                        if self.strike_count <= 2:
                            logger.info("Model did not provide a valid answer or request.")
                            conversation.append(
                                {
                                    "role": "user", 
                                    "content": "Please provide the final answer in [ANSWER: LETTER) answer] specifying the answer choice letter you picked (A, B, C, D, E, or F) or ask for files with [REQUEST: filename.extension]. Make sure you have both the opening an closing brackets."
                                }
                            )
                        else:
                            logger.info("Model failed to provide a valid answer or request after 3 attempts.")
                            self.strike_count = 0
                            question_end_time = time.time()
                            correct = False
                            self.chat_history.append({
                                "question": question,
                                "answer": expected_answer,
                                "response": response,
                                "correct": correct,
                                "files_accessed": all_files_accessed_for_question,
                                "files_hallucinated": all_halucinated_files,
                                "question_time": question_end_time - question_start_time,
                            })
                            break
                        
        # Log the entire conversation
        self.chat_history.append({
            "conversation": conversation
        })
        self._store_log(case_id)
        return self.chat_history


    def _store_log(self, case_id):
        """
        Store the chat history in a JSON file.
        """
        log_path = self.output_dir / f"{case_id}_chatlog_{int(time.time())}.json"
        with open(log_path, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
        logger.info(f"Chat history saved to {log_path}")
