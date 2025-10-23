from pathlib import Path
import base64
from neurips25.eval.base_hf_eval import BaseHuggingFaceEval
from loguru import logger
import openai

class GPT4oEval(BaseHuggingFaceEval):
    def __init__(self, model_name, hf_token=None, openai_token=None):
        super().__init__(model_name, hf_token)
        self.openai_token = openai_token
        self.load_client()

    def load_model(self):
        pass

    def load_processor(self):
        pass

    def load_client(self):
        """
        Initialize the OpenAI client with the API key.
        """
        self.client = openai.Client(api_key=self.openai_token)

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
                    content_list.append({"type": "image_url", "image_url": {"url": data_url}})
            content_list.append({"type": "text", "text": message["content"]})
            prompt.append({"role": role, "content": content_list})
        return prompt

    @logger.catch
    def process_text(self, messages):
        """
        Process the text and vision inputs.
        """
        messages = self.convert_to_chat_format(messages)
        return messages

    def generate_response(self, inputs):
        """
        Generate a response using the chat client based on the processed inputs.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=inputs,
            max_tokens=1024,
            temperature=0.2,
        )
        return response

    def decode_response(self, response):
        """
        Decode the response from the model into a human-readable text.
        """
        output_text = response.choices[0].message.content
        return output_text
