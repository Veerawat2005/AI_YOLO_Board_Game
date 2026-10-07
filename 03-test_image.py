from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

# โมเดลล่าสุด (best.pt ที่รากโฟลเดอร์ คือ train-6 แล้ว)
model = YOLO(str(BASE / "best.pt"))

# รูปอยู่ในโฟลเดอร์เดียวกับสคริปต์
image = BASE / "3c143777-e201-4d7b-a687-83f53d382f3e.jpg"

results = model.predict(
    str(image),
    conf=0.8,
    save=True
)

results[0].show()

# สรุปสิ่งที่ตรวจเจอ
for box in results[0].boxes:
    print(model.names[int(box.cls)], f"{float(box.conf):.2f}")