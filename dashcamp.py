from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch

# Check if a CUDA-enabled GPU is available, otherwise default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device_status_text = f"Processing on: {device.upper()}"
print(device_status_text) # Print the status to the console as well

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("bikes.mp4")
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video


model = YOLO("../Yolo-Weights/yolov8n.pt")

# Move the model to the selected device (GPU or CPU)
model.to(device)


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break # Break the loop if the video has ended

    # When calling the model, you can also explicitly set the device, e.g., model(img, stream=True, device=device)
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate and print FPS
    fps_val = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps_val:.2f}")

    # --- ADDED CODE: Display FPS and Device Status ---
    # Display FPS on the top right
    cv2.putText(img, f"FPS: {int(fps_val)}", (img.shape[1] - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Display the device status on the top left
    cv2.putText(img, device_status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
