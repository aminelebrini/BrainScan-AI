import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from ultralytics import YOLO

class ModelBuilder:
    def __init__(self):
        self.model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        self.yolo_model = YOLO("yolov8n.pt")

    def build_model(self):
        self.model.trainable = False

        model = models.Sequential([
            self.model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(
                256, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']

        )

        return model

    def build_yolo_model(self):
        return self.yolo_model
    def validate_yolo_model(self, yolo_model_version):
        self.yolo_model = YOLO(yolo_model_version)
        return self.yolo_model