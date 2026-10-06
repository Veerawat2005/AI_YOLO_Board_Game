from ultralytics import YOLO

# โหลดโมเดลที่ผ่านการฝึก (Trained Model)
model = YOLO(r"E:\จารย์รุจิ\Ai_yolo\runs\detect\train-6\weights\best.pt")

# นำโมเดลไปทดสอบกับรูปภาพ
results = model.predict(
    r"E:\จารย์รุจิ\Ai_yolo\dataset\images\val\20261002_124932_001.jpg",
    conf=0.5,
    save=True
)

# แสดงผลลัพธ์การตรวจจับของรูปภาพแรก
results[0].show()