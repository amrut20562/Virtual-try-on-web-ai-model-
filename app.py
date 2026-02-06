import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import traceback
from werkzeug.middleware.proxy_fix import ProxyFix

from pipeline import VirtualTryOnPipeline

# ----------------------------
# Flask setup
# ----------------------------


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PERSON_DIR = os.path.join(UPLOAD_DIR, "person")
GARMENT_DIR = os.path.join(UPLOAD_DIR, "garment")

RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

os.makedirs(PERSON_DIR, exist_ok=True)
os.makedirs(GARMENT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,
    x_proto=1,
    x_host=1,
    x_port=1
)


# ----------------------------
# Load model ONCE
# ----------------------------
print("🔄 Loading Virtual Try-On pipeline...")
vton_pipeline = VirtualTryOnPipeline()
print("✅ Pipeline loaded")

# ----------------------------
# Routes
# ----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tryon", methods=["POST"])
def tryon():
    try:
        # -------- Validation --------
        if "person_image" not in request.files:
            return jsonify({"success": False, "error": "Person image missing"}), 400

        if "garment_image" not in request.files:
            return jsonify({"success": False, "error": "Garment image missing"}), 400

        garment_type = request.form.get("garment_type")
        if garment_type not in ["shirt", "pants"]:
            return jsonify({"success": False, "error": "Invalid garment type"}), 400

        person_file = request.files["person_image"]
        garment_file = request.files["garment_image"]

        # -------- Save uploads --------
        uid = uuid.uuid4().hex

        person_path = os.path.join(PERSON_DIR, f"{uid}_person.png")
        garment_path = os.path.join(GARMENT_DIR, f"{uid}_garment.png")

        person_file.save(person_path)
        garment_file.save(garment_path)

        # -------- Load images --------
        person_img = Image.open(person_path).convert("RGB")
        garment_img = Image.open(garment_path).convert("RGB")

        # -------- Run model --------
        output_img = vton_pipeline.run(
            person_img=person_img,
            garment_img=garment_img,
            garment_type=garment_type
        )

        # -------- Save result --------
        result_filename = f"{uid}_result.png"
        result_path = os.path.join(RESULT_DIR, result_filename)
        output_img.save(result_path)

        # -------- Return URL --------
        image_url = f"/static/results/{result_filename}"

        return jsonify({
            "success": True,
            "image_url": image_url
        })

    except Exception as e:
        traceback.print_exc() 
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----------------------------
# Run app
# ----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )
