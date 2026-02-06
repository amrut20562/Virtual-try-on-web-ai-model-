import os

DEVICE = "cuda"

HF_TOKEN = os.getenv("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")  # <-- ADD THIS

SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_openpose"
HUMAN_PARSING_MODEL = "mattmdjaga/segformer_b2_clothes"

IP_ADAPTER_MODEL = "h94/IP-Adapter"
IP_ADAPTER_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

IMAGE_SIZE = 512
