import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector

class PoseExtractor:
    def __init__(self):
        # ❌ NO token here (controlnet-aux does not support it)
        self.detector = OpenposeDetector.from_pretrained(
            "lllyasviel/ControlNet"
        )

    def extract(self, image: Image.Image):
        image_np = np.array(image)
        pose = self.detector(image_np)
        return pose
