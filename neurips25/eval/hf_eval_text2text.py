import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from neurips25.eval.base_hf_eval import BaseHuggingFaceEval
from loguru import logger
from vllm import LLM, SamplingParams
from neurips25.tools import search_pubmed

class BaseTextHFEval(BaseHuggingFaceEval):
    def __init__(self, tools=False, *args, **kwargs):
        """
        Initialize the BaseTextHFEval class with model name and system prompt.
        """
        super().__init__(*args, **kwargs)
        self.load_model()
        self.tools = []
        if tools:
            self.tools = [search_pubmed]

    def load_processor(self):
        pass

    def load_model(self):
        """
        Load the model from the specified model name.
        """
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token

    def convert_to_chat_format(self, messages):
        """
        Convert the input messages to the chat format required by the model.
        """
        prompt = []
        for message in messages:
            new_message = {"role": message["role"], "content": ""}
            if "files" in message:
                files_concat = ""
                for file in message["files"]:
                    files_concat += f"[{file['type']}]" # define what to do here with files
                new_message["content"] += files_concat
            new_message["content"] += message["content"]
            prompt.append(new_message)
        return prompt

    @logger.catch
    def process_text(self, messages):
        """
        Process the input text and prepare it for the model.
        """
        tools = self.tools if self.tools else None
        messages = self.convert_to_chat_format(messages)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools,
        )
        inputs = self.tokenizer(
            text,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        return inputs

    def generate_response(self, inputs):
        """
        Generate a response from the model based on the inputs.
        """
        with torch.no_grad():
            response = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
            ).cpu()
        return (response, inputs["input_ids"].shape[1])

    def decode_response(self, response):
        """
        Decode the generated response to a human-readable format.
        """
        (response, inputs) = response
        output_text = self.tokenizer.batch_decode(
            response[:,inputs:], skip_special_tokens=True
        )
        return output_text[0]

class BaseTextVLLMEval(BaseHuggingFaceEval):
    def __init__(self, tools=False, use_all_but_last_device=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_all_but_last_device = use_all_but_last_device
        self.load_model()
        self.tools = []
        if tools:
            self.tools = [search_pubmed]

    def load_processor(self):
        pass

    def load_model(self):
        """
        Load the model from the specified model name.
        """
        to_quantize = ["Qwen/Qwen3-32B", "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-235B-A22B", "mistralai/Mixtral-8x22B-Instruct-v0.1"]
        self.model = LLM(
            model=self.model_name, 
            trust_remote_code=True,
            tokenizer_mode="mistral" if self.model_name.startswith("mistralai/Mistral-7B-Instruct") else "auto",
            config_format="mistral" if self.model_name.startswith("mistralai/Mistral-7B-Instruct") else "hf",
            load_format="mistral" if self.model_name.startswith("mistralai/") else "auto",
            dtype=torch.bfloat16 if self.model_name in to_quantize else "auto",
            quantization="bitsandbytes" if self.model_name in to_quantize else None,
            enforce_eager=False,
            rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768} if self.model_name == "Qwen/Qwen3-235B-A22B" else None,
            max_num_seqs=1,
            # max_model_len=70000 if self.model_name.endswith("11B-Vision-Instruct") else 120000,
            tensor_parallel_size=max(torch.cuda.device_count() - 1, 1) if self.use_all_but_last_device else torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
        )
        self.sampling_params = SamplingParams(
            max_tokens=1536,
            temperature=0.0,
        )

    def convert_to_chat_format(self, messages):
        """
        Convert the input messages to the chat format required by the model.
        """
        prompt = []
        for message in messages:
            new_message = {"role": message["role"], "content": ""}
            if "files" in message:
                files_concat = ""
                for file in message["files"]:
                    files_concat += f"[{file['type']}]" # define what to do here with files
                new_message["content"] += files_concat
            new_message["content"] += message["content"]
            prompt.append(new_message)
        return prompt

    @logger.catch
    def process_text(self, messages):
        """
        Process the input text and prepare it for the model.
        """
        # tools = self.tools if self.tools else None
        return self.convert_to_chat_format(messages)

    def generate_response(self, inputs):
        """
        Generate a response from the model based on the inputs.
        """
        return self.model.chat(messages=inputs, sampling_params=self.sampling_params)

    def decode_response(self, response):
        """
        Decode the generated response to a human-readable format.
        """
        return response[0].outputs[0].text