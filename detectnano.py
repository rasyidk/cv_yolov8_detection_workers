import cv2
import time
from ultralytics import YOLO

# Load model YOLOv8 nano custom
model = YOLO("yolov8n_custom.pt")  # Pastikan file ini ada di direktori yang benar

# Buka video
cap = cv2.VideoCapture("vid1.mp4")  # Gunakan video yang kamu miliki

# Atur jumlah frame yang akan dilewati
frame_skip = 6  # Lompat setiap 2 frame

frame_count = 0  # Counter frame
total_inference_time = 0  # Variabel untuk menyimpan total inference time
processed_frames = 0  # Jumlah frame yang dideteksi

while cap.isOpened():
    start_time = time.time()  # Mulai hitung total waktu eksekusi per iterasi

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # Tambah counter frame

    # Hanya deteksi pada frame ganjil (1, 3, 5, 7, ...)
    if frame_count % frame_skip == 1:
        inference_start = time.time()  # Mulai hitung waktu inferensi
        results = model(frame)
        inference_end = time.time()  # Akhiri hitung waktu inferensi

        inference_time = inference_end - inference_start  # Hitung total inference time
        total_inference_time += inference_time  # Tambahkan ke total inference time
        processed_frames += 1  # Tambah jumlah frame yang diproses

        print(f"Frame {frame_count}: Inference Time = {inference_time:.4f} seconds")

        # Render hasil deteksi pada frame
        annotated_frame = results[0].plot()

        # Tampilkan hasil
        cv2.imshow('YOLOv8 Detection', annotated_frame)

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
