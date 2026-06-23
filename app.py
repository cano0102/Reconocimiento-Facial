from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from analyzer import analyze_from_bytes
import traceback
import logging
import imghdr

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------
# Ruta principal: Ahora renderiza el archivo desde la carpeta templates
# ---------------------------------------------------------
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
        return jsonify({"error": "Formato de archivo no permitido"}), 400

    filename = secure_filename(file.filename)

    try:
        file_bytes = file.read()

        # Validación extra real del contenido (no solo extensión)
        if imghdr.what(None, h=file_bytes) is None:
            return jsonify({"error": "El archivo no es una imagen válida"}), 400

        result = analyze_from_bytes(file_bytes)

        # Validación defensiva
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Formato inválido de respuesta del analizador")

        faces, img_b64 = result

        return jsonify({
            "faces": faces or [],
            "annotated_image": img_b64
        })

    except Exception as e:
        logger.error("Error en /analyze: %s", traceback.format_exc())
        return jsonify({
            "error": "Error interno en el análisis",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    print("App corriendo en http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)