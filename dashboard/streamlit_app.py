from pathlib import Path

import streamlit as st
from PIL import Image

from inference import (
    CLASSIFIER_PATH,
    DETECTOR_PATH,
    detect_objects,
    load_metrics,
    classify_image,
)

st.set_page_config(page_title="BrainScan AI", page_icon="🧠", layout="wide")
st.title("BrainScan AI")
st.write("Classification et détection de tumeurs cérébrales sur image IRM.")


@st.cache_resource
def load_models(classifier_path: str, detector_path: str):
    import tensorflow as tf
    from ultralytics import YOLO

    classifier = tf.keras.models.load_model(classifier_path, compile=False)
    detector = YOLO(detector_path)
    return classifier, detector


def main() -> None:
    uploaded_file = st.file_uploader(
        "Charger une image IRM",
        type=["jpg", "jpeg", "png"],
    )
    confidence = st.slider("Seuil de confiance YOLO", 0.0, 1.0, 0.25, 0.05)

    if not uploaded_file:
        st.info("Chargez une image pour lancer l'inférence.")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            classifier, detector = load_models(str(CLASSIFIER_PATH), str(DETECTOR_PATH))
            classifications = classify_image(classifier, image)
            annotated_image, detections = detect_objects(detector, image, confidence)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            st.error(f"Inférence impossible : {error}")
            return

        classification_column, detection_column = st.columns(2)
        with classification_column:
            st.subheader("Classification")
            best = max(classifications, key=lambda item: float(item["probability"]))
            st.metric("Catégorie prédite", str(best["category"]))
            st.dataframe(classifications, use_container_width=True, hide_index=True)

        with detection_column:
            st.subheader("Détections")
            st.image(annotated_image, caption="Boîtes de détection", use_container_width=True)
            if detections:
                st.dataframe(detections, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune boîte au-dessus du seuil choisi.")

    st.subheader("Métriques du modèle")
    try:
        metrics = load_metrics()
    except (OSError, ValueError) as error:
        st.error(f"Lecture des métriques impossible : {error}")
    else:
        if metrics:
            st.dataframe(
                [{"metric": key, "value": value} for key, value in metrics.items()],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Aucune métrique sauvegardée. Exécutez le DAG d'entraînement.")


if __name__ == "__main__":
    main()
