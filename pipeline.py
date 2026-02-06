import os
import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel,StableDiffusionControlNetInpaintPipeline
from PIL import Image
from config import *
from human_parsing import HumanParser
from controlnet_utils import PoseExtractor
from face_utils import FaceExtractor


class VirtualTryOnPipeline:
    def __init__(self):

        
        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL,
            token=HF_TOKEN,
            torch_dtype=torch.float16
        ).to(DEVICE)

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL,
            controlnet=self.controlnet,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
        ).to(DEVICE)

        # IP-Adapter (correct usage)
        self.pipe.load_ip_adapter(
            IP_ADAPTER_MODEL,
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
            image_encoder_folder="image_encoder",
            token=HF_TOKEN
        )

        self.pipe.set_ip_adapter_scale(1.0)

        self.parser = HumanParser(HUMAN_PARSING_MODEL, DEVICE)
        self.pose_extractor = PoseExtractor()
        self.face_extractor = FaceExtractor(HUMAN_PARSING_MODEL, DEVICE)


        os.makedirs("debug", exist_ok=True)

    def _save(self, img, name):
        img.save(f"debug/{name}")

    def _overlay_mask(self, image, mask):
        image_np = np.array(image).copy()
        mask_np = np.array(mask)

        # red overlay
        image_np[mask_np > 0] = (
            0.6 * image_np[mask_np > 0] + np.array([255, 0, 0]) * 0.4
        )

        return Image.fromarray(image_np.astype(np.uint8))

    def _merge_face(self, original_img, generated_img, face_mask):
        orig = np.array(original_img).astype(np.float32)
        gen = np.array(generated_img).astype(np.float32)
        mask = np.array(face_mask).astype(np.float32) / 255.0

        # Expand mask to 3 channels
        mask = np.stack([mask] * 3, axis=-1)

        merged = orig * mask + gen * (1 - mask)
        merged = np.clip(merged, 0, 255).astype(np.uint8)

        return Image.fromarray(merged)


    def run(self, person_img, garment_img, garment_type="shirt"):
        person_img = person_img.resize((IMAGE_SIZE, IMAGE_SIZE))

        # ---- DEBUG 1: inputs
        self._save(person_img, "01_person.png")
        #self._save(shirt_img, "02_shirt.png")

        # ---- DEBUG 2: human parsing mask
        if garment_type == "shirt":
            mask = self.parser.get_upper_cloth_mask(person_img)
        elif garment_type == "pants":
            mask = self.parser.get_pants_mask(person_img)
        else:
            raise ValueError("garment_type must be 'shirt' or 'pants'")

        
        self._save(mask, "03_mask_raw.png")

        mask_overlay = self._overlay_mask(person_img, mask)
        self._save(mask_overlay, "04_mask_overlay.png")

        assert mask.size == person_img.size, \
            f"Mask size {mask.size} != Image size {person_img.size}"


        # ---- DEBUG 3: masked input image
        masked_person = person_img.copy()
        masked_person_np = np.array(masked_person)
        masked_person_np[np.array(mask) > 0] = 0
        masked_person = Image.fromarray(masked_person_np)
        self._save(masked_person, "05_masked_person.png")
        

        # ---- DEBUG 4: ControlNet pose
        pose = self.pose_extractor.extract(person_img)
        self._save(pose, "06_pose.png")

        # ---- extract original face
        orig_face_img, face_mask = self.face_extractor.extract_face(person_img)

        # ---- DIFFUSION
        result = self.pipe(
            prompt= "a person wearing clothing, realistic fabric texture, natural folds, "
                    "correct garment fit on the body, consistent lighting, high detail, "
                    "photorealistic, clean edges",
            negative_prompt="distorted face, altered face, deformed face, unrealistic face, "
                    "extra arms, extra legs, missing limbs, fused limbs, "
                    "wrong garment, incorrect clothing, mismatched outfit, "
                    "wrong color, recolored fabric, color shift, washed out colors, "
                    "flat texture, plastic texture, blurry fabric, low detail, "
                    "floating clothing, detached clothing, incorrect shadows, "
                    "artifacts, noise, jpeg artifacts, blur",
            image=person_img,
            mask_image=mask,
            control_image=pose,
            ip_adapter_image=garment_img,
            num_inference_steps=30,
            guidance_scale=8.5,
            strength=0.8 #0.65 0.90.7 0.75 0.85 0.8 0.6
        ).images[0]

        # ---- HARD FACE MERGE (v7.7 FINAL STEP)
        result = self._merge_face(orig_face_img, result, face_mask)

        self._save(result, "07_result.png")
        self._save(garment_img, "08_ip_adapter_reference.png")


        return result

