import torch
import numpy as np
from PIL import Image
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

class Conch:

    def __init__(self, device: str, post_process_size=1024) -> None:

        # Used to bypass library limitation as our WSI can be quite big
        Image.MAX_IMAGE_PIXELS = None
        
        self.device = device
        self.model, self.preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", force_image_size=post_process_size, device=self.device)
        self.tokenizer = get_tokenizer()

    
    def image_to_text_retrieval(self, image_path: str, prompts: list[str]) -> tuple[str, list[float]]:
        """
        Finds the most relevant text prompt for a given image by comparing image and text embeddings.

        Args:
            image_path (str): The path to a histopathology image file in SVS format.
            prompts (list[str]): A list of text prompts to compare against the image

        Returns:
            tuple[str, list[float]]: A tuple where the first element is the predicted class (most relevant prompt)
                                    and the second element is a list of similarity scores between the image and each prompt
        """
        image = Image.open(image_path)

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        tokenized_prompts = tokenize(texts=prompts, tokenizer=self.tokenizer).to(self.device)

        with torch.inference_mode():
            image_embedings = self.model.encode_image(image_tensor)
            text_embedings = self.model.encode_text(tokenized_prompts)
            sim_scores = (image_embedings @ text_embedings.T * self.model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()

        predicted_class = prompts[sim_scores.argmax()]

        # Return predicted class and all class probabilities
        return predicted_class, sim_scores


    def image_to_image_retrieval(self, image_path: str, other_image_paths: list[str]) -> tuple[np.ndarray, list[float]]:
        """
        Finds the most similar image from a list of images given a reference image.

        Args:
            image_path (str): The path to a histopathology image file in SVS format.
            other_image_paths (list[str]): A list of paths to histopathology image files in SVS format.

        Returns:
            tuple[str, list[float]]: A tuple where the first element is the path to the most similar image
                                            and the second element is a list of similarity scores between the reference image
                                            and each image in the list
        """
        image = Image.open(image_path)

        other_images = []
        for image_path in other_image_paths:
            other_images.append(Image.open(image_path))

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        other_images_tensors = torch.concat([self.preprocess(img).unsqueeze(0).to(self.device) for img in other_images], axis=0)

        with torch.inference_mode():
            image_embedding = self.model.encode_image(image_tensor)
            other_images_embeddings = self.model.encode_image(other_images_tensors)
            sim_scores = (image_embedding @ other_images_embeddings.T * self.model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()

        most_similar_image = other_image_paths[sim_scores.argmax()]
        return most_similar_image, sim_scores


    def text_to_image_retrieval(self, prompt: str, other_image_paths: list[str]) -> tuple[str, list[float]]:
        """
        Finds the most relevant image based on a given text prompt from a list of images.

        Args:
            prompt (str): The text prompt used to query for the most relevant image
            other_image_paths (list[str]): A list of images to compare to the text prompt

        Returns:
            tuple[str, list[float]]: A tuple where the first element is the path to the most relevant image to the prompt
                                            and the second element is a list of similarity scores between the prompt and each image
        """
        other_images = []
        for image_path in other_image_paths:
            other_images.append(Image.open(image_path))

        tokenized_prompt = tokenize(texts=[prompt], tokenizer=self.tokenizer).to(self.device)
        other_images_tensors = torch.concat([self.preprocess(img).unsqueeze(0).to(self.device) for img in other_images], axis=0)

        with torch.inference_mode():
            text_embedding = self.model.encode_text(tokenized_prompt)
            other_images_embeddings = self.model.encode_image(other_images_tensors)
            sim_scores = (text_embedding @ other_images_embeddings.T * self.model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()

        most_similar_image = other_image_paths[sim_scores.argmax()]
        return most_similar_image, sim_scores
