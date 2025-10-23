from neurips25.eval import *
from omegaconf import OmegaConf
import os

def get_model(model_name):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
        return MistralVLLMEval(model_name=model_name), "mistralsmall"
    if model_name == "meta-llama/Llama-3.2-90B-Vision-Instruct":
        return LlamaVLLMEval(model_name=model_name), "llama90b"
    elif "gpt-4o" in model_name.lower():
        conf = OmegaConf.load("neurips25/configs/base.yaml")
        openai_token = conf.openai_token
        return GPT4oEval(model_name=model_name, openai_token=openai_token), "gpt-4o"
    elif "Qwen2.5-VL" in model_name:
        return Qwen25VLEval(model_name=model_name), "qwen2.5-vl"
    elif "Qwen3" in model_name:
        return BaseTextVLLMEval(model_name=model_name), "qwen3"
    elif "Llama-3" in model_name:
        return BaseTextVLLMEval(model_name=model_name), "llama-3"
    elif "gemma-3" in model_name.lower():
        return Gemma3Eval(model_name=model_name), "gemma-3"
    elif "internvl3" in model_name.lower():
        return InternVLEval(model_name=model_name), "internvl3"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    