import cv2
from ultralytics import YOLO


def main():
    model = YOLO(r"E:\จารย์รุจิ\Ai_yolo\runs\detect\train-6\weights\best.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    print("กด 'q' เพื่อออกจากโปรแกรม")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(
            source=frame,
            device=0,        # ใช้ GPU (ถ้าไม่มี CUDA ให้เปลี่ยนเป็น 'cpu')
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()
        cv2.imshow("YOLO26 Real-time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()