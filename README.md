# human detection with yolo v8

---

# Real-Time Human and Object Detection Using YOLO and Webcam

This project demonstrates real-time human and object detection using a webcam and the YOLO algorithm. The YOLO model used in this project is `yolov8n.pt` from the `ultralytics` package. The code captures video from the webcam, processes each frame to detect humans and other objects, and displays the results in real-time.

## Requirements

To run this project, you need the following dependencies:

- Python 3.6 or higher
- OpenCV
- Ultralytics YOLO

You can install the required packages using the following commands:

```bash
pip install opencv-python ultralytics
```

## Usage

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/real-time-human-object-detection.git
   cd real-time-human-object-detection
   ```

2. **Download the YOLOv8 Weights**

   Ensure you have the YOLOv8 weights file (`yolov8n.pt`). If not, download it from the official [Ultralytics YOLO repository](https://github.com/ultralytics/yolov5/releases).

3. **Run the Script**

   Run the Python script to start the real-time human and object detection:

   ```bash
   python camera_face_detection.py
   ```

4. **Exit the Program**

   Press 'q' to exit the program.

## Code Explanation

The main script `camera_face_detection.py` performs the following steps:

1. **Import Necessary Libraries**

   ```python
   import os
   import cv2
   from ultralytics import YOLO
   ```

2. **Initialize the YOLO Model**

   Load the YOLO model using the `ultralytics` package.

   ```python
   model = YOLO('yolov8n.pt')
   ```

3. **Initialize the Webcam**

   Capture video from the webcam.

   ```python
   cap = cv2.VideoCapture(0)
   ```

4. **Process Each Frame**

   Capture each frame, perform object detection, and draw bounding boxes around detected objects, including humans.

   ```python
   while True:
       ret, frame = cap.read()
       if not ret:
           break

       results = model(frame)
       for result in results:
           boxes = result.boxes
           for box in boxes:
               x1, y1, x2, y2 = map(int, box.xyxy[0])
               class_id = int(box.cls[0])
               confidence = float(box.conf[0])

               if class_id < len(class_names):
                   label = class_names[class_id]
                   color = (0, 255, 0) if label == 'person' else (0, 0, 255)
                   cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                   cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

       cv2.imshow('Human and Object Detection', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()
   ```

## Troubleshooting

- **No video feed**: Ensure your webcam is properly connected and accessible.
- **Missing dependencies**: Ensure all required packages are installed using `pip install -r requirements.txt`.
- **YOLO weights file not found**: Ensure `yolov8n.pt` is in the correct directory or provide the correct path.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5) for the YOLO implementation.
- [OpenCV](https://opencv.org/) for the computer vision library.
