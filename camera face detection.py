import os
import cv2
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # You can change to yolov8m.pt, yolov8l.pt, etc. based on your preference

# Class names corresponding to YOLO model
class_names = model.names  # Get class names from the model

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Processing detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Check if the class_id is within the range of class_names
            if class_id < len(class_names):
                label = class_names[class_id]

                # Draw bounding boxes and labels on the frame
                if label == 'person':
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    # Display the resulting frame
    cv2.imshow('Human Detection', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
