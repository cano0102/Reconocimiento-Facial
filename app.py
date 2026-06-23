from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from analyzer import analyze_from_bytes
import traceback
import logging
from PIL import Image
import io

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Archivo vacío"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Formato no permitido"}), 400

    try:
        file_bytes = file.read()

        # Validación real de imagen
        try:
            Image.open(io.BytesIO(file_bytes)).verify()
        except Exception:
            return jsonify({"error": "Archivo no es una imagen válida"}), 400

        result = analyze_from_bytes(file_bytes)

        faces, img_b64 = result

        return jsonify({
            "faces": faces or [],
            "annotated_image": img_b64
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Error interno",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run()