import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Initialize YOLOv10 model
model = YOLO('yolov10n.yaml')  # Ensure you have the appropriate YOLOv10 configuration file

# Define the data configuration
data_config = {
    'train': 'path/to/train/images',
    'val': 'path/to/val/images',
    'nc': 1,  # Number of classes
    'names': ['person']  # Class names
}

# Train the model
results = model.train(data=data_config, epochs=50, imgsz=640)

# Extract training and validation metrics
metrics = results.metrics
epochs = range(len(metrics['train/loss']))

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, metrics['train/loss'], label='Train Loss')
plt.plot(epochs, metrics['val/loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

# Plot validation precision, recall, mAP, and F1 score
plt.subplot(1, 2, 2)
plt.plot(epochs, metrics['val/precision'], label='Precision')
plt.plot(epochs, metrics['val/recall'], label='Recall')
plt.plot(epochs, metrics['val/mAP_0.5'], label='mAP@0.5')
plt.plot(epochs, metrics['val/F1'], label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.title('Validation Metrics')

plt.tight_layout()
plt.show()
