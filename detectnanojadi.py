import cv2
import time
from ultralytics import YOLO

# Load model YOLOv8 nano custom
model = YOLO("yolov8n_custom.pt")  # Pastikan file ini ada di direktori yang benar

# Buka video
cap = cv2.VideoCapture("vid1.mp4")  # Gunakan video yang kamu miliki

# Atur jumlah frame yang akan dilewati
frame_skip = 2  # Lompat setiap 2 frame

frame_count = 0  # Counter frame

while cap.isOpened():
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # Tambah counter frame

    # Hanya deteksi pada frame ganjil (1, 3, 5, 7, ...)
    if frame_count % frame_skip == 1:
        # Jalankan deteksi dengan YOLOv8
        results = model(frame)

        # Render hasil deteksi pada frame
        annotated_frame = results[0].plot()

        # Tampilkan hasil
        cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
