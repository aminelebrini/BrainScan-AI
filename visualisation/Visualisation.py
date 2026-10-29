import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


class Visualisation:
    def __init__(self):
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    def display_extracted_image(self, data_images):

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for image in data_images:
           for i, j in zip(range(2), range(2)):
                axes[i, j].imshow(image)
                axes[i, j].set_title(image.name)
                axes[i, j].axis('off')

        plt.show()

    def display_image_by_opencv(self, images_path, labels):

            
        for image_path, label_path in zip(images_path[:4], labels[:4]):
            class_name = "Inconnu"
            labels_info = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
    
                for line in lines:
                    line = line.strip().split()
                    if len(line) == 5:
                        class_id, x_center, y_center, width, height = line
                        labels_info.append((int(class_id), float(x_center), float(y_center), float(width), float(height)))
                if labels_info:
                    class_id = labels_info[0][0]
                    class_name = self.class_names[class_id]

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            h, w, _ = image.shape

            for box in labels_info:
                class_id, x_center, y_center, width, height = box
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, self.class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.title(f"Image: {class_name}")
            plt.axis('off')
            plt.show()

    def display_image_after_augmentation(self, augmented_images):
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

        for image, labels_batch in augmented_images.take(1):
        
            single_image = image[0].numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(single_image)
            plt.title(f"Label: {class_names[labels_batch[0].numpy()]}")
            plt.axis('off')
            plt.show()
            break

    @staticmethod
    def _load_history(history_path):
        try:
            with history_path.open(encoding="utf-8") as history_file:
                history = json.load(history_file)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"History file not found: {history_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON history file: {history_path}") from error

        if not isinstance(history, dict):
            raise ValueError(f"History must be a JSON object: {history_path}")

        return history

    def display_training_results(
        self,
        feature_extraction_history=Path(
            "results/feature_extraction/history_data_model_feature_extraction.json"
        ),
        fine_tuning_history=Path("results/fin-tuining/history_data_model.json"),
        output_dir=Path("results/visualizations"),
    ):
        feature_history = self._load_history(feature_extraction_history)
        fine_tuning_history_data = self._load_history(fine_tuning_history)

        required_metrics = {"accuracy", "loss", "val_accuracy", "val_loss"}
        for model_name, history in (
            ("Feature extraction", feature_history),
            ("Fine-tuning", fine_tuning_history_data),
        ):
            missing_metrics = required_metrics - history.keys()
            if missing_metrics:
                missing = ", ".join(sorted(missing_metrics))
                raise ValueError(f"{model_name} history is missing metrics: {missing}")

        sns.set_theme(style="whitegrid", context="notebook")
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        for axis, metrics, title in (
            (axes[0], ("accuracy", "val_accuracy"), "Accuracy comparison"),
            (axes[1], ("loss", "val_loss"), "Loss comparison"),
        ):
            for model_name, history in (
                ("Feature extraction", feature_history),
                ("Fine-tuning", fine_tuning_history_data),
            ):
                for metric in metrics:
                    values = history[metric]
                    sns.lineplot(
                        x=range(1, len(values) + 1),
                        y=values,
                        marker="o",
                        label=f"{model_name} - {metric}",
                        ax=axis,
                    )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Value")
            axis.grid(alpha=0.25)

        figure.suptitle(
            "Feature extraction vs fine-tuning",
            fontsize=16,
            fontweight="bold",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "training_comparison.png"
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.show()
        plt.close(figure)
        print(f"Training visualization saved to {output_path}")
        return output_path

    def display_yolo_results(self, yolo_results_path):
        plt.figure(figsize=(6, 6))
        for image_path, model_name in yolo_results_path:
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title(f"YOLO Result Confusion Matrix: {model_name}")
            plt.axis('off')
            plt.show()