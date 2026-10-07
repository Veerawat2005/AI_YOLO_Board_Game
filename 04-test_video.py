from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

# โมเดลอยู่ข้างสคริปต์ (best.pt ที่ราก Ai_yolo)
model = YOLO(str(BASE / "best.pt"))

# วิดีโอทดสอบ ต้องอยู่ข้างสคริปต์ หรือใส่ path เต็ม
video_to_test = BASE / "test_vid2.mp4"

if not video_to_test.exists():
    raise SystemExit(f"ไม่พบไฟล์วิดีโอ: {video_to_test}")

results = model.predict(
    source=str(video_to_test),
    save=True,       # บันทึกวิดีโอผลลัพธ์
    show=True,       # แสดงผลระหว่างประมวลผล (กด q เพื่อหยุด)
    conf=0.8,
    device=0,        # ใช้ GPU (ถ้าไม่มี CUDA ให้เปลี่ยนเป็น "cpu")
    stream=True,     # ประมวลผลทีละเฟรม ไม่กิน RAM กับวิดีโอยาว
)

# ใช้ stream=True ต้องวนลูปเพื่อให้ทำงานจริง
for _ in results:
    pass

print("ทดสอบเสร็จสิ้น! ดูวิดีโอผลลัพธ์ได้ที่โฟลเดอร์ runs/detect/predict...")