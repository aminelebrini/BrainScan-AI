import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from pathlib import Path
import json
from ultralytics import YOLO

train_images_path = Path("/kaggle/input/datasets/aminelebrini/dataset-data/train/images")
train_labels_path = Path("/kaggle/input/datasets/aminelebrini/dataset-data/train/labels")
val_images_path = Path("/kaggle/input/datasets/aminelebrini/dataset-data/val/images")
val_labels_path = Path("/kaggle/input/datasets/aminelebrini/dataset-data/val/labels")
test_images_path = Path("/kaggle/input/datasets/aminelebrini/dataset-data/test/images")
test_labels_path = Path("/kaggle/input/datasets/aminelebrini/dataset-data/test/labels")

# model = ResNet50(
#             weights='imagenet',
#             include_top=False,
#             input_shape=(224, 224, 3)
#         )

# def load_data_set(images_path, labels_path, batch_size=32, shuffle=True):

#         data_set = tf.data.Dataset.from_tensor_slices((images_path, labels_path))

#         if shuffle:
#             data_set = data_set.shuffle(buffer_size=len(images_path))

#         data_set = data_set.batch(batch_size)
#         data_set = data_set.prefetch(tf.data.AUTOTUNE)

#         return data_set

# def resize_and_normalize_image(images_path, target_size=(224, 224)):
#         class_names = ['Glioma', 'Meningioma', 'NoTumor', 'Pituitary']
#         labels_path = images_path.parent / "labels"
#         normalized_images = []
#         labels = []
#         for image_path in images_path.glob("*"):
#             if image_path.is_file() and image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

#                 with Image.open(image_path) as img:
#                     img = img.convert("RGB")
#                     image = img.resize(target_size)
#                     image = np.array(image, dtype=np.float32) / 255.0

#                 label_file = labels_path / f"{image_path.stem}.txt"

#                 if label_file.exists():
#                  with open(label_file, 'r') as f:
#                         line = f.readline().strip()
#                         if line:

#                             class_id = int(float(line.split()[0]))

#                             normalized_images.append(image)
#                             labels.append(class_id)
#                 else:
#                     print(f"Warning: Label file for {image_path.name} not found!")

#         return np.array(normalized_images, dtype=np.float32), np.array(labels, dtype=np.int32)


# def data_augmentation(images):
#             Data_augmentation = tf.keras.Sequential([
#                 tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#                 tf.keras.layers.RandomRotation(0.2),
#                 tf.keras.layers.RandomZoom(0.1),
#             ])

#             augmented_images = images.map(lambda x, y: (Data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)

#             return augmented_images

# # def build_model(model):
# #         model.trainable = False

# #         model = models.Sequential([
# #             model,
# #             layers.GlobalAveragePooling2D(),
# #             layers.Dropout(0.3),
# #             layers.Dense(
# #                 256,
# #                 activation='relu',
# #                 kernel_regularizer=tf.keras.regularizers.l2(0.001)
# #             ),
# #             layers.Dropout(0.3),
# #             layers.Dense(4, activation='softmax')
# #         ])

# #         model.compile(
# #             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
# #             loss='sparse_categorical_crossentropy',
# #             metrics=['accuracy']

# #         )

# #         return model

# normalized_train_images, normalized_train_label = resize_and_normalize_image(train_images_path)
# normalized_val_images, normalized_val_label = resize_and_normalize_image(val_images_path)
# normalized_test_images, normalized_test_label = resize_and_normalize_image(test_images_path)

# train_data_set = load_data_set(normalized_train_images, normalized_train_label, batch_size=32)

# val_data_set = load_data_set(normalized_val_images,normalized_val_label, batch_size=32)
# augmented_train_images = data_augmentation(train_data_set)

# model = build_model(model)
# model.summary()

# save_path_file = Path("models/best_model/resnet50_feature_extraction_best_version.keras")

# callbacks = [
#   tf.keras.callbacks.EarlyStopping(
#       monitor="val_loss",
#       patience=6,
#       restore_best_weights=True,
#   ),
#   tf.keras.callbacks.ModelCheckpoint(
#       save_path_file,
#       monitor='val_accuracy',
#       save_best_only=True

#   ),
#   tf.keras.callbacks.ReduceLROnPlateau(
#       monitor="val_loss",
#       factor=0.2,
#       min_lr=1e-6,
#       patience=2
#   ),
# ]

# history = model.fit(augmented_train_images, epochs=30, validation_data=val_data_set, callbacks=callbacks)

# history_data = history.history

# yolo_model = YOLO("yolov8n.pt")
yolo_model = YOLO("runs/detect/yolo_best_v-2/weights/best.pt")

# yolo_model.train(
#         data="data.yaml",
#         split="train",
#         epochs=15,
#         batch=10,
#         imgsz=640,
#         device=0,
#         name="yolo_best_v"
#     )


metrics = yolo_model.val(data="data.yaml", split="test", device=0)

precision = metrics.box.mp
recall = metrics.box.mr
f1_score = 0.0
if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0

metrics_data = {
    "mAP50-95": float(metrics.box.map),
    "mAP50": float(metrics.box.map50),
    "Precision": float(precision),
    "Recall": float(recall),
    "F1-Score": float(f1_score)
}

yolo_metrics_path = "evaluation_metrics.json"
with open(yolo_metrics_path, 'w') as f:
    json.dump(metrics_data, f, indent=4)

print(f"📁 Metrics successfully saved to JSON file: {yolo_metrics_path}")
print("\n" + "="*40)
print("📊 RESULTATS D L-EVALUATION & F1-SCORE:")
print("="*40)
print(f"{metrics_data}")
print("="*40 + "\n")

# result = yolo_model.predict(
#     source="/kaggle/input/datasets/aminelebrini/dataset-data/test/images",
#     conf=0.50,
#     iou=0.50,
#     device=0,
#     save=True,
#     save_txt=True,
#     save_conf=True
# )
print("Predictions & Visualizations saved successfully in runs/detect/predict! 🎉")



# with open("history_data_model_feature_extraction.json", "w") as file:
#     json.dump(history_data, file)

# print("Training history saved to training_history.json successfully!")

# yolo_model.predict(
#     source="/kaggle/input/datasets/aminelebrini/dataset-data/test/images",
#     conf=0.50,
#     device=0,
#     save=True,
# )
print("Training Completed Successfully ! 🎉")

