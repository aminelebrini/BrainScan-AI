from datetime import datetime, timedelta
import tensorflow as tf
import json
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
from models.ModelBuilder import ModelBuilder
from src.DataTraitement import DataTraitement

train_traitement = DataTraitement()

def execute_data_exploration():
    train_traitement.extract_zip_data()
    train_dir = Path("data/extracted_data/Train")
    train_images, train_labels = train_traitement.data_explorator(train_dir)

    train_traitement.check_corrupted_annotations()
    train_images, train_labels = train_traitement.remove_clean_corrompus_annotations()

    val_traitement = DataTraitement()
    val_dir = Path("data/extracted_data/Val")
    val_images, val_labels = val_traitement.data_explorator(val_dir)
    
    val_traitement.check_corrupted_annotations()
    val_images, val_labels = val_traitement.remove_clean_corrompus_annotations()

    return train_images, train_labels, val_images, val_labels

def prepare_dataset_split():

    train_images, train_labels, val_images, val_labels = execute_data_exploration()
    train_traitement.prepare_dataset_split(train_images, train_labels, val_images, val_labels)

    train_images_path = Path("data/splited_data/train/images")
    train_labels_path = Path("data/splited_data/train/labels")
    val_images_path = Path("data/splited_data/val/images")
    val_labels_path = Path("data/splited_data/val/labels")
    test_images_path = Path("data/splited_data/test/images")
    test_labels_path = Path("data/splited_data/test/labels")

    normalized_train_images, normalized_train_label = train_traitement.resize_and_normalize_image(train_images_path)
    normalized_val_images, normalized_val_label = train_traitement.resize_and_normalize_image(val_images_path)
    normalized_test_images, normalized_test_label = train_traitement.resize_and_normalize_image(test_images_path)

    return normalized_train_images, normalized_train_label, normalized_val_images, normalized_val_label, normalized_test_images, normalized_test_label


def data_augmentation():

    normalized_train_images, normalized_train_label, normalized_val_images, normalized_val_label, normalized_test_images, normalized_test_label = prepare_dataset_split()
    
    train_data_set = train_traitement.load_data_set(normalized_train_images, normalized_train_label, batch_size=32)
    
    val_data_set = train_traitement.load_data_set(normalized_val_images,normalized_val_label, batch_size=32)
    augmented_train_images = train_traitement.data_augmentation(train_data_set)

    return augmented_train_images, normalized_val_images, normalized_val_label, normalized_test_images, normalized_test_label, val_data_set


def build_and_train_model():

    augmented_train_images, normalized_val_images, normalized_val_label, normalized_test_images, normalized_test_label , val_data_set = data_augmentation()

    model_builder = ModelBuilder()
    model = model_builder.build_model()

    save_path_file = Path("models/best_model/resnet50_fine_tuining_best_version.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            save_path_file,
            monitor='val_accuracy',
            save_best_only=True
    
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            min_lr=1e-6,
            patience=2
        ),
            
    ]
    
    EPOCHS = 30
    history = model.fit(
        augmented_train_images,
        validation_data=val_data_set,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=False
    )

    history_data = history.history
    
    fine_tuning_history_path = Path("results/fin-tuining/history_data_model.json")
    fine_tuning_history_path.parent.mkdir(parents=True, exist_ok=True)
    with fine_tuning_history_path.open("w", encoding="utf-8") as file:
        json.dump(history_data, file)

def test_validate_model():

    augmented_train_images, normalized_val_images, normalized_val_label, normalized_test_images, normalized_test_label , val_data_set = data_augmentation()

    test_data_set = train_traitement.load_data_set(normalized_test_images, normalized_test_label, batch_size=32)
    model = Path("models/best_model/resnet50_fine_tuining_best_version.keras")
    model = tf.keras.models.load_model(model)

    y_test = []
    y_pred = []

    for images, labels in test_data_set:
        predictions = model.predict(images)
        predicted_labels = tf.argmax(predictions, axis=1)
        y_test.extend(labels.numpy())
        y_pred.extend(predicted_labels.numpy())
    
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print("Evaluation Metrics:")
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(cm)

def train_yolo_model():
    model_builder = ModelBuilder()
    model = model_builder.build_yolo_model()

    model.train(
        data="data.yaml",
        epochs=30,
        batch=16,
        imgsz=640,
        device="cpu",
        name="models/yolo_best_v"
    )

    best_model_path = Path("models/yolo_best_v/weights/best.pt")
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"YOLO training completed but the best model was not found at "
            f"{best_model_path}."
        )

    model = model_builder.validate_yolo_model(str(best_model_path))
    metrics = model.val(data="data.yaml", split="test", device="cpu")

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1_value = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    metrics_data = {
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_value,
    }

    metrics_path = Path("results/visualizations/evaluation_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_data, file, indent=4)

    print(f"YOLO evaluation metrics saved to {metrics_path}: {metrics_data}")
    return metrics_data


default_args = {
    "owner": "brainscan-ai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="brain_scan_training_pipeline",
    default_args=default_args,
    description="Train and evaluate the BrainScan classification and detection models.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["brainscan", "training"],
) as brain_scan_training_dag:
    train_classifier = PythonOperator(
        task_id="train_classifier",
        python_callable=build_and_train_model,
    )

    evaluate_classifier = PythonOperator(
        task_id="evaluate_classifier",
        python_callable=test_validate_model,
    )

    train_detector = PythonOperator(
        task_id="train_detector",
        python_callable=train_yolo_model,
    )

    train_classifier >> evaluate_classifier >> train_detector

    
