import cv2
from dotenv import dotenv_values
from ultralytics import YOLO
from ultralytics.engine.results import Results


def main():
    dotenv = dotenv_values(".env")

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

    try:
        print("Started running... (Press q or ESC to stop)")
        while True:
            _, frame = cap.read()
            results: list[Results] = model(frame, verbose=False, classes=filtered_obj)
            annotated_frame = results[0].plot()
            cv2.imshow("Detected frame", annotated_frame)

            if cv2.waitKey(1) in (ord("q"), 27):
                break

    finally:
        cap.release()


if __name__ == "__main__":
    main()
