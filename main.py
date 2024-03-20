import csv
import json
import msvcrt
import time
from datetime import datetime

import cv2
from dotenv import dotenv_values
from ultralytics import YOLO
from ultralytics.engine.results import Results


def get_current_time():
    return int(time.time())


def create_video(filename: str):
    return cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        14.0,
        (1280, 720),
    )


def save_frequency_data(filename: str, data: tuple[int, int]):
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


def main():
    dotenv = dotenv_values(".env")

    csv_file = f"{get_current_time()}.csv"
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "frequency"])

    use_trained = True
    use_open_images_v7 = False
    filtered_obj = 0 if use_trained else 381 if use_open_images_v7 else 0
    model = YOLO(
        "yolov8n-trained.pt"
        if use_trained
        else "yolov8n-oiv7.pt"
        if use_open_images_v7
        else "yolov8n.pt"
    )

    CCTV_IP = "192.168.1.36"
    cap = cv2.VideoCapture(
        f"rtsp://{dotenv['USERNAME']}:{dotenv['PASSWORD']}@{CCTV_IP}/stream1"
    )
    if not cap.isOpened():
        raise IOError("Cannot open cctv")

    filename: str | None = None
    out: cv2.VideoWriter | None = None
    last_detected: int | None = None

    last_save: int = get_current_time()
    last_obj_count: int = 0

    try:
        print("Started recording... (Press q or ESC to stop)")
        while True:
            _, frame = cap.read()
            results: list[Results] = model(frame, verbose=False, classes=filtered_obj)
            obj_ls: list = json.loads(results[0].tojson())
            obj_count = len(obj_ls)
            max_obj_count = max(obj_count, last_obj_count)

            if get_current_time() - last_save == 1:
                save_frequency_data(csv_file, (last_save, max_obj_count))
                last_save = get_current_time()
                last_obj_count = 0

            if obj_ls:
                last_detected = get_current_time()
                if out is None:
                    time_str = datetime.fromtimestamp(last_detected)
                    print("Found Person on", time_str)
                    filename = f"{time_str.strftime('%Y%m%d_%H-%M-%S')}.mp4"
                    out = create_video(filename)
            elif last_detected is not None:
                if get_current_time() > last_detected + 1:
                    print("Save video:", filename)
                    out = None
                    filename = None
                    last_detected = None

            if out is not None:
                out.write(results[0].plot())

            if msvcrt.kbhit():
                if msvcrt.getch() in (b"q", b"\x1b"):
                    print("Stopping...")
                    break
    finally:
        cap.release()
        if out is not None:
            out.release()


if __name__ == "__main__":
    main()
