import cv2
import time
from ultralytics import YOLO

# Load model YOLOv8 nano custom
model = YOLO("yolov8n_custom.pt")  # Pastikan model tersedia

# Gunakan kamera sebagai sumber video
cap = cv2.VideoCapture("vid1.mp4")

cap.set(3, 320)  # Lebar
cap.set(4, 240)

frame_skip = 6  # Deteksi setiap 6 frame
frame_count = 0
total_inference_time = 0
processed_frames = 0

while cap.isOpened():
    start_time = time.time()  # Hitung waktu eksekusi per frame

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Hanya deteksi pada frame tertentu
    if frame_count % frame_skip == 1:
        inference_start = time.time()
        results = model(frame)
        inference_end = time.time()

        inference_time = inference_end - inference_start
        total_inference_time += inference_time
        processed_frames += 1

        print(f"Frame {frame_count}: Inference Time = {inference_time:.4f} seconds")

        # Render hasil deteksi
        annotated_frame = results[0].plot()

        # Tampilkan hasil
        cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Hitung rata-rata inference time
if processed_frames > 0:
    avg_inference_time = total_inference_time / processed_frames
    print(f"\nTotal Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Inference Time per Frame: {avg_inference_time:.4f} seconds")
