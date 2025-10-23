from neurips25.eval.base_hf_eval import BaseHuggingFaceEval
from qwen_vl_utils import process_vision_info
from loguru import logger
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
import torch

class Qwen25VLEval(BaseHuggingFaceEval):
    def __init__(self, model_name, hf_token=None, use_last_device_for_eval=False):
        super().__init__(model_name, hf_token)
        self.use_last_device_for_eval = use_last_device_for_eval
        self.load_model()
        self.load_processor()
    
    def load_model(self):
        """
        Load the model from the specified model name.
        """
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto" if not self.use_last_device_for_eval else f"cuda:{torch.cuda.device_count() - 1}",
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
        )

    def load_processor(self):
        """
        Load the processor from the specified model name.
        """
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def convert_to_chat_format(self, messages):
        """
        Convert the input messages to the chat format required by the model.
        """
        prompt = []
        for message in messages:
            new_message = {"role": message["role"], "content": []}
            if "files" in message:
                for file in message["files"]:
                    new_message["content"].append({"type": "image", "image": file})
            new_message["content"].append({"type": "text", "text": message["content"]})
            prompt.append(new_message)
        return prompt
        
    @logger.catch
    def process_text(self, messages):
        messages = self.convert_to_chat_format(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda", dtype=torch.bfloat16)
        return inputs
    
    def generate_response(self, inputs):
        """
        Generate a response from the model based on the inputs.
        """
        with torch.inference_mode():
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=1536)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return generated_ids_trimmed
    
    def decode_response(self, generated_ids):
        """
        Decode the generated response into human-readable text.
        """
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return output_text[0]