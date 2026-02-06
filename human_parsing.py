import torch
import cv2

import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from config import HF_TOKEN


class HumanParser:
    def __init__(self, model_name, device):
        self.extractor = SegformerImageProcessor.from_pretrained(model_name, token=HF_TOKEN)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, token=HF_TOKEN).to(device)
        self.device = device

    def get_upper_cloth_mask(self, image: Image.Image):
        inputs = self.extractor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()

        # -----------------------------
        # Shirt region
        # -----------------------------
        shirt_mask = np.zeros_like(seg, dtype=np.uint8)
        shirt_mask[seg == 4] = 255  # upper clothes

        # -----------------------------
        # Arm occlusion (v7.7)
        # -----------------------------
        arm_mask = np.zeros_like(seg, dtype=np.uint8)
        arm_mask[seg == 9] = 255   # left arm
        arm_mask[seg == 10] = 255  # right arm

        # Expand arms slightly to ensure clean occlusion
        kernel = np.ones((5, 5), np.uint8)
        arm_mask = cv2.dilate(arm_mask, kernel, iterations=1)

        # Remove arms from shirt region
        shirt_mask[arm_mask > 0] = 0

        # -----------------------------
        # Final mask
        # -----------------------------
        mask = Image.fromarray(shirt_mask)
        mask = mask.resize(image.size, resample=Image.NEAREST)
        return mask


          
    def get_pants_mask(self, image: Image.Image):
      inputs = self.extractor(images=image, return_tensors="pt").to(self.device)

      with torch.no_grad():
          outputs = self.model(**inputs)

      seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()

      # Pants region
      mask = np.zeros_like(seg, dtype=np.uint8)
      mask[seg == 6] = 255  # pants

      # Protect face + hair
      protect_mask = np.zeros_like(seg, dtype=np.uint8)
      protect_mask[seg == 11] = 255  # face
      protect_mask[seg == 2] = 255   # hair

      kernel = np.ones((11, 11), np.uint8)
      protect_mask = cv2.dilate(protect_mask, kernel, iterations=1)

      mask[protect_mask > 0] = 0

      mask = Image.fromarray(mask)
      mask = mask.resize(image.size, resample=Image.NEAREST)
      return mask




