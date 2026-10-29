from src.DataTraitement import DataTraitement
from visualisation.Visualisation import Visualisation
from models.ModelBuilder import ModelBuilder
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import json
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

def main():
    # 1. Instance w Traitement dyal Train
    train_traitement = DataTraitement()
    train_traitement.extract_zip_data()
    
    train_dir = Path("data/extracted_data/Train")
    train_images, train_labels = train_traitement.data_explorator(train_dir)

    train_traitement.check_corrupted_annotations()
    train_images, train_labels = train_traitement.remove_clean_corrompus_annotations()

    print("Train count:", len(train_images))
    print(train_images, train_labels)

    val_traitement = DataTraitement()
    val_dir = Path("data/extracted_data/Val")
    val_images, val_labels = val_traitement.data_explorator(val_dir)

    val_traitement.check_corrupted_annotations()
    val_images, val_labels = val_traitement.remove_clean_corrompus_annotations()

    print("Validation count:", len(val_images))
    print(val_images, val_labels)

    visualisation = Visualisation()
    visualisation.display_image_by_opencv(train_images[:4], train_labels[:4])

    train_traitement.prepare_dataset_split(train_images, train_labels, val_images, val_labels)

    print("Dataset split completed successfully.")

    print("Starting Redomentionement and Normalisation !")

    train_images_path = Path("data/splited_data/train/images")
    train_labels_path = Path("data/splited_data/train/labels")
    val_images_path = Path("data/splited_data/val/images")
    val_labels_path = Path("data/splited_data/val/labels")
    test_images_path = Path("data/splited_data/test/images")
    test_labels_path = Path("data/splited_data/test/labels")

    data_traitement = DataTraitement()
    normalized_train_images, normalized_train_label = data_traitement.resize_and_normalize_image(train_images_path)
    normalized_val_images, normalized_val_label = data_traitement.resize_and_normalize_image(val_images_path)
    normalized_test_images, normalized_test_label = data_traitement.resize_and_normalize_image(test_images_path)

    # print(f"Number of loaded train images: {len(normalized_train_images)}")

    print("Redomentionement and Normalisation completed successfully !")

    # print("Starting Visualisation of Normalized Images !")

    # visualisation = Visualisation()
    # visualisation.display_image_by_opencv(normalized_train_images[:4], normalized_train_label[:4])

    # print("Visualisation of Normalized Images completed successfully !")

    print("starting data augmentation !")

    train_data_set = data_traitement.load_data_set(normalized_train_images, normalized_train_label, batch_size=32)

    val_data_set = data_traitement.load_data_set(normalized_val_images,normalized_val_label, batch_size=32)
    augmented_train_images = data_traitement.data_augmentation(train_data_set)

    print("Data augmentation completed successfully !")

    visualisation.display_image_after_augmentation(augmented_train_images)

    model_builder = ModelBuilder()
    model = model_builder.build_model()
    model.summary()
    print("Model Built Successfully !")

    print("making callbacks functions")

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
    # history = model.fit(
    #     augmented_train_images,
    #     validation_data=val_data_set,
    #     epochs=EPOCHS,
    #     callbacks=callbacks,
    #     shuffle=False
    # )

    print("Training Completed Successfully ! 🎉")

    # # history_data = history.history

    # fine_tuning_history_path = Path("results/fin-tuining/history_data_model.json")
    # fine_tuning_history_path.parent.mkdir(parents=True, exist_ok=True)
    # with fine_tuning_history_path.open("w", encoding="utf-8") as file:
    #     json.dump(history_data, file)

    visualisation.display_training_results()

    # print("Strating Evaluation of the model on the test set !")

    # test_data_set = data_traitement.load_data_set(normalized_test_images, normalized_test_label, batch_size=32)

    # y_test = []
    # y_pred = []

    # for images, labels in test_data_set:
    #     predictions = model.predict(images)
    #     predicted_labels = tf.argmax(predictions, axis=1)
    #     y_test.extend(labels.numpy())
    #     y_pred.extend(predicted_labels.numpy())

    # recall = recall_score(y_test, y_pred, average='weighted')
    # precision = precision_score(y_test, y_pred, average='weighted')
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # cm = confusion_matrix(y_test, y_pred)

    # print("Evaluation Metrics:")
    # print("Recall:", recall)
    # print("Precision:", precision)
    # print("F1 Score:", f1)
    # print("Confusion Matrix:")
    # print(cm)

    print("Evaluation of the model on the test set completed successfully !")

    # print("Starting Object Detection using YOLOv8 !")

    # model = model_builder.build_yolo_model()

    # model.train(
    #     data="data.yaml",
    #     epochs=30,
    #     batch=16,
    #     imgsz=640,
    #     device="cpu",
    #     name="models/yolo_best_v"
    # )

    # model = model_builder.validate_yolo_model("models/yolo_best_v/weights/best.pt")
    # metrics = model.val(data="data.yaml", split="test", device=0)

    # precision = metrics.box.mp
    # recall = metrics.box.mr
    # f1_score = 0.0
    # if (precision + recall) > 0:    
    #     f1_score = 2 * (precision * recall) / (precision + recall)
    # else:
    #     f1_score = 0.0

    # metrics_data = {
    #     "mAP50-95": float(metrics.box.map),
    #     "mAP50": float(metrics.box.map50),
    #     "Precision": float(precision),
    #     "Recall": float(recall),
    #     "F1-Score": float(f1_score)
    # }

    yolo_v8n_metrics_path = "results/visualizations/evaluation_metrics.json"
    yolo_v11n_metrics_path = "results/visualizations/evaluation_v11_metrics.json"
    # with open(yolo_metrics_path, 'w') as f:
    #     json.dump(metrics_data, f, indent=4)

    with open(yolo_v8n_metrics_path, "r", encoding="utf-8") as file:
        metrics_data = json.load(file)

    with open(yolo_v11n_metrics_path, "r", encoding="utf-8") as file:
        metrics_v11_data = json.load(file)

    print(f"📁 Metrics successfully saved to JSON file: {yolo_v8n_metrics_path}")
    print("\n" + "="*40)
    print("📊 RESULTATS D L-EVALUATION & F1-SCORE:")
    print("="*40)
    print(f"{metrics_data}")
    print("="*40 + "\n")

    print(f"📁 Metrics successfully saved to JSON file: {yolo_v11n_metrics_path}")
    print("\n" + "="*40)
    print("📊 RESULTATS D L-EVALUATION & F1-SCORE:")
    print("="*40)
    print(f"{metrics_v11_data}")
    print("="*40 + "\n")

    # result = model.predict(
    #     source="/kaggle/input/datasets/aminelebrini/dataset-data/test/images",
    #     conf=0.50,
    #     iou=0.50,
    #     device=0,
    #     save=True,
    #     save_txt=True,
    #     save_conf=True
    # )
    print("Predictions & Visualizations saved successfully 🎉")

    print("Object Detection using YOLOv8 completed successfully ! 🎉")

    print("Starting Visualisation of YOLO Versions Results !")
    yolo_results_path = [
        [Path("images/confusion_matrix_yolo_v11n.png"), "YOLOV11N"],[Path("images/confusion_matrix.png"), "YOLOV8"],
        [Path("images/Te-gl_0052.jpg"), "Glioma V11n Detection"],[Path("images/val_batch0_pred.jpg"), "Meningioma V8n Detection"]
    ]
    visualisation.display_yolo_results(yolo_results_path)

if __name__ == "__main__":
    main()