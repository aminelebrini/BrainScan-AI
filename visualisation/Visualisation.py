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


