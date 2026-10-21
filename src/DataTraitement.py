from pathlib import Path
import zipfile
from sklearn.model_selection import train_test_split
# import tensorflow as tf
import os
from PIL import Image
class DataTraitement:
    def __init__(self):
        self.images = []
        self.labels = []
        self.class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

    def extract_zip_data(self):
        zip_data_path = Path("data/archive_irm.zip")
        extract_path = Path("data/extracted_data")
        if zip_data_path.exists():
            if not extract_path.exists():
                print(f"Extracting {zip_data_path} to {extract_path}...")
                with zipfile.ZipFile(zip_data_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"Extraction completed. Files are available at {extract_path}.")
            else:
                print(f"{extract_path} is already exist !")
        else:
            print(f"Zip file {zip_data_path} does not exist. Please check the path and try again.")

    def data_explorator(self, data_images):

        if not data_images.exists():
            print(f"Directory {data_images} does not exist. Please check the path and try again.")
            return

        for class_dir in data_images.iterdir():
            if class_dir.is_dir():
                images_dir = class_dir / "images"
                labels_dir = class_dir / "labels"

            if images_dir.exists() and labels_dir.exists():
                for image_path in images_dir.iterdir():
                    if image_path.is_file():
                        label_path = labels_dir / f"{image_path.stem}.txt"
                    
                    if label_path.exists():
                        self.images.append(image_path)
                        self.labels.append(label_path)
                    
        
        return self.images, self.labels

    def check_corrupted_annotations(self):
        corrompus = []
        
        for image_path in self.images:
            with Image.open(image_path) as img:

                try:
                    img.verify()

                except Exception as e:
                    print(f"corrompus detected {image_path}")
                    corrompus.append(image_path)

        if len(corrompus) == 0:
            print("corrompus don't exist !")
        else:
            print(f"number of corrompus {len(corrompus)}")
        
        errors_count = 0
        for label_path in self.labels:
            if not label_path.exists():
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:
                    print(f"Inconsistency detected in {label_path}: Expected 5 values, but got {len(parts)}.")
                    errors_count += 1
                    continue

                try:       
                    class_id, x_center, y_center, width, height = parts
                    class_id = int(class_id)
                    x_center = float(x_center)
                    y_center = float(y_center)
                    width = float(width)
                    height = float(height)
                except ValueError:
                    print(f"Inconsistency detected in {label_path}: Unable to convert values to the expected types.")
                    errors_count += 1
                    continue

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Inconsistency detected in {label_path}: Bounding box values are out of range [0, 1].")
                    errors_count += 1
                if class_id not in [0,1,2,3]:
                    print(f"Inconsistency detected in {label_path}: Class ID {class_id} is not valid. Expected values are 0, 1, 2, or 3.")
                    errors_count += 1

            if errors_count == 0:
                print(f"No inconsistencies found in {label_path}.")
            else:
                print(f"Total inconsistencies found in {label_path}: {errors_count}.")
                
    def remove_clean_corrompus_annotations(self):
        valid_images = []
        valid_labels = []
        removed_images = 0

        for image_path , label_path in zip(self.images, self.labels):
        
            is_corrupted = False
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                is_corrupted = True
        
            if is_corrupted:
                if image_path.exists():
                    os.remove(image_path)
                if label_path.exists():
                    os.remove(label_path)
                removed_images +=1
                print("image corrompus has been successfuly removed !")
            else:
                valid_images.append(image_path)
                valid_labels.append(label_path)

        if removed_images > 0:
            print(f"Removed {removed_images} corrupted images and their corresponding labels.")

        self.images = valid_images
        self.labels = valid_labels
        
        return self.images, self.labels


    def prepare_dataset_split(self,train_images, train_labels, val_images, val_labels):
        train_dataset_path = Path("data/splited_data/")

        train_images_path = train_dataset_path / "train/images"
        train_labels_path = train_dataset_path / "train/labels"
        val_images_path = train_dataset_path / "val/images"
        val_labels_path = train_dataset_path / "val/labels"
        test_images_path = train_dataset_path / "test/images"
        test_labels_path = train_dataset_path / "test/labels"

        train_images_path.mkdir(parents=True, exist_ok=True)
        train_labels_path.mkdir(parents=True, exist_ok=True)
        val_images_path.mkdir(parents=True, exist_ok=True)
        val_labels_path.mkdir(parents=True, exist_ok=True)
        test_images_path.mkdir(parents=True, exist_ok=True)
        test_labels_path.mkdir(parents=True, exist_ok=True)

                    
        if len(train_images) > 1:

            train_dataset , test_dataset, train_dataset_labels, test_dataset_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

            for image_path, label_path in zip(train_dataset, train_dataset_labels):
                img_pa = Path(image_path)
                label_pa = Path(label_path)
                    
                destination_image_path = train_images_path / img_pa.name
                destination_label_path = train_labels_path / f"{label_pa.stem}.txt"

                if img_pa.exists():
                    destination_image_path.write_bytes(img_pa.read_bytes())
                if label_pa.exists():
                    destination_label_path.write_text(label_pa.read_text())
                else:
                    destination_label_path.touch()

            print(f"Training dataset split completed. {len(train_dataset)} images and labels saved to {train_images_path} and {train_labels_path}.")
            print("Starting test dataset split...")

            for image_path, label_path in zip(test_dataset, test_dataset_labels):
                img_pa = Path(image_path)
                label_pa = Path(label_path)
                                    
                destination_image_path = test_images_path / img_pa.name
                destination_label_path = test_labels_path / f"{label_pa.stem}.txt"
                
                if img_pa.exists():
                    destination_image_path.write_bytes(img_pa.read_bytes())
                if label_pa.exists():
                    destination_label_path.write_text(label_pa.read_text())
                else:
                    destination_label_path.touch()

            print(f"Test dataset split completed. {len(test_dataset)} images and labels saved to {test_images_path} and {test_labels_path}.")

            print("Starting validation dataset split...")

            for image_path, label_path in zip(val_images, val_labels):
                img_pa = Path(image_path)
                label_pa = Path(label_path)

                destination_image_path = val_images_path / img_pa.name
                destination_label_path = val_labels_path / f"{label_pa.stem}.txt"


                if img_pa.exists():
                    destination_image_path.write_bytes(img_pa.read_bytes())
                if label_pa.exists():
                    destination_label_path.write_text(label_pa.read_text())
                else:
                    destination_label_path.touch()

            print(f"Validation dataset split completed. {len(val_images)} images and labels saved to {val_images_path} and {val_labels_path}.")

        

        
            

