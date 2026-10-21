from src.DataTraitement import DataTraitement
from visualisation.Visualisation import Visualisation
from pathlib import Path

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
if __name__ == "__main__":
    main()