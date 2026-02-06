import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from config import HF_TOKEN


class FaceExtractor:
    def __init__(self, model_name, device):
        self.extractor = SegformerImageProcessor.from_pretrained(
            model_name, token=HF_TOKEN
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, token=HF_TOKEN
        ).to(device)
        self.device = device

    def extract_face(self, image: Image.Image):
        """
        Returns:
        - face_image (PIL)
        - face_mask (PIL, soft-edged)
        """
        inputs = self.extractor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()

        # Face label = 11 (verified earlier)
        face_mask = np.zeros_like(seg, dtype=np.uint8)
        face_mask[seg == 11] = 255

        # Resize to image size
        face_mask = Image.fromarray(face_mask)
        face_mask = face_mask.resize(image.size, Image.NEAREST)

        # Convert to numpy for smoothing
        face_mask_np = np.array(face_mask)

        # Expand slightly + feather edges
        kernel = np.ones((9, 9), np.uint8)
        face_mask_np = cv2.dilate(face_mask_np, kernel, iterations=1)
        face_mask_np = cv2.GaussianBlur(face_mask_np, (21, 21), 0)

        face_mask = Image.fromarray(face_mask_np)

        return image, face_mask
