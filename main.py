from PIL import Image
from pipeline import VirtualTryOnPipeline

from huggingface_hub import login
import os

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

if __name__ == "__main__":
    person = Image.open("inputs/person3.png").convert("RGB")

    # ---- USER CHOICE ----
    GARMENT_TYPE = "pants"  # "shirt" or "pants"

    if GARMENT_TYPE == "shirt":
        garment = Image.open("inputs/cloth3.jpg").convert("RGB")
    elif GARMENT_TYPE == "pants":
        garment = Image.open("inputs/pants.webp").convert("RGB")


    vton = VirtualTryOnPipeline()
    output = vton.run(person, garment, garment_type=GARMENT_TYPE)


    os.makedirs("output", exist_ok=True)

    output.save("output/tryon_result.png")
    print("✅ Try-on image saved to output/tryon_result.png")
