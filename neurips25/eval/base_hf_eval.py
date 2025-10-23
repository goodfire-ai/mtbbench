import torch
import torch.nn as nn
import json
from huggingface_hub import login
from abc import ABC, abstractmethod


class BaseHuggingFaceEval(ABC):
    def __init__(self, model_name, hf_token=None):
        self.model_name = model_name
        if hf_token is not None:
            login(token=hf_token)

    @abstractmethod
    def load_model(self):
        """
        Load the model from the specified model name.
        """
        pass

    @abstractmethod
    def load_processor(self):
        """
        Load the processor from the specified model name.
        """
        pass


    @abstractmethod
    def convert_to_chat_format(self, messages):
        """
        Convert the input messages to the chat format required by the model.
        """
        pass

    @abstractmethod
    def process_text(self, messages):
        """
        Process the input text and prepare it for the model.
        """
        pass

    @abstractmethod
    def generate_response(self, inputs):
        """
        Generate a response from the model based on the inputs.
        """
        pass

    @abstractmethod
    def decode_response(self, response):
        """
        Decode the generated response to a human-readable format.
        """
        pass

    def evaluate(self, messages):
        """
        Evaluate the model with the given messages.
        """
        inputs = self.process_text(messages)
        response = self.generate_response(inputs)
        decoded_response = self.decode_response(response)
        return decoded_response

