from pathlib import Path
import base64
from neurips25.eval.base_hf_eval import BaseHuggingFaceEval
from qwen_vl_utils import process_vision_info
from loguru import logger
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
import torch

class InternVLEval(BaseHuggingFaceEval):
    def __init__(self, model_name, hf_token=None):
        super().__init__(model_name, hf_token)
        self.load_model()
        self.load_processor()
    
    def load_model(self):
        """
        Load the model from the specified model name.
        """
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

    def load_processor(self):
        """
        Load the processor from the specified model name.
        """
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _image_to_data_url(self, path: str) -> str:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image does not exist: {path}")

        with path.open("rb") as fp:
            encoded = base64.b64encode(fp.read()).decode("utf-8")

        if path.suffix.lower() == ".png":
            return f"data:image/png;base64,{encoded}"
        elif path.suffix.lower() == ".jpg" or path.suffix.lower() == ".jpeg":
            return f"data:image/jpeg;base64,{encoded}"
        else:
            raise ValueError(f"Unsupported image format: {path.suffix}. Supported formats are .png, .jpg, .jpeg.")
    
    def convert_to_chat_format(self, messages):
        """
        Convert the input messages to the chat format required by OpenAI.
        """
        prompt = []
        for message in messages:
            role = message["role"]
            content_list = []
            if "files" in message:
                for file in message["files"]:
                    data_url = self._image_to_data_url(file)
                    content_list.append({"type": "image", "url": data_url})
            content_list.append({"type": "text", "text": message["content"]})
            prompt.append({"role": role, "content": content_list})
        return prompt
        
    # @logger.catch
    def process_text(self, messages):
        messages = self.convert_to_chat_format(messages)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        return inputs
    
    def generate_response(self, inputs):
        """
        Generate a response from the model based on the inputs.
        """
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            return (generated_ids, inputs["input_ids"].shape[1])
    
    def decode_response(self, generated_ids):
        """
        Decode the generated response into human-readable text.
        """
        (generate_ids, inputs) = generated_ids
        output_text = self.processor.decode(generate_ids[0, inputs :], skip_special_tokens=True)
        return output_text