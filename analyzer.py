from deepface import DeepFace  # type: ignore[import]

import cv2
import numpy as np
import base64
from pathlib import Path
import tempfile
import os


# ================== ETIQUETAS ==================
GENDER_LABELS_ES = {
    "Man": "Hombre",
    "Woman": "Mujer",
}

RACE_LABELS_ES = {
    "asian": "Asiático",
    "indian": "Indio",
    "black": "Afrodescendiente",
    "white": "Caucásico",
    "middle eastern": "Oriente Medio",
    "latino hispanic": "Latino / Hispano",
}


# ================== UTIL ==================
def convert_numpy_to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(i) for i in obj]
    return obj


# ================== ANALISIS ==================
def analyze_image(image_path: str | Path) -> list[dict]:

    try:
        results = DeepFace.analyze(
            img_path=str(image_path),
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )

        if isinstance(results, dict):
            results = [results]

        if not results:
            raise ValueError("No se detectó ningún rostro.")

    except Exception as e:
        raise ValueError(f"Error al analizar la imagen: {e}")

    faces = []

    for r in results:
        r_clean = convert_numpy_to_python(r)

        dominant_gender = r_clean.get("dominant_gender", "Unknown")
        dominant_race = r_clean.get("dominant_race", "Unknown")

        gender_scores = r_clean.get("gender", {})
        race_scores = r_clean.get("race", {})

        # ================== FIX EDAD ==================
        age_raw = r_clean.get("age", None)

        try:
            edad = int(float(age_raw)) if age_raw is not None else None
            print(f"DEBUG - Edad procesada: {edad}")
        except Exception:
            edad = None

        faces.append({
            "genero": GENDER_LABELS_ES.get(dominant_gender, dominant_gender),
            "genero_confianza": round(float(gender_scores.get(dominant_gender, 0)), 1),
            "raza_dominante": RACE_LABELS_ES.get(dominant_race, dominant_race),

            "edad_estimada": edad,

            "emocion": r_clean.get("dominant_emotion", "Neutral"),

            "emociones_detalle": [
                {"nombre": k, "porcentaje": round(float(v), 1)}
                for k, v in r_clean.get("emotion", {}).items()
            ],

            "razas_detalle": {
                k: round(float(v), 1)
                for k, v in race_scores.items()
            },

            "region": r_clean.get("region", {})
        })

    return faces


# ================== BYTES ==================
def analyze_from_bytes(image_bytes: bytes) -> tuple[list[dict], str]:

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)

    try:
        faces = analyze_image(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    annotated = draw_annotations(img.copy(), faces)

    _, buffer = cv2.imencode(".jpg", annotated)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return faces, img_b64


# ================== DIBUJO ==================
def draw_annotations(img: np.ndarray, faces: list[dict]) -> np.ndarray:

    for i, face in enumerate(faces, start=1):

        r = face.get("region", {})

        if not r:
            continue

        x = int(r.get("x", 0))
        y = int(r.get("y", 0))
        w = int(r.get("w", 0))
        h = int(r.get("h", 0))

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 100), 2)
        cv2.putText(
            img,
            f"Rostro {i}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return img