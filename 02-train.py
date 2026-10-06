from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolo26n.pt')

    results = model.train(
        data=r"E:\จารย์รุจิ\Ai_yolo\dataset\data.yaml",
        epochs=150,
        imgsz=640,
        optimizer="MuSGD",
        device=0,

        # ===== เพิ่ม 3 บรรทัดนี้ (แก้ crash) =====
        workers=0,      # เดิมใช้ค่า default (8) ทำให้ worker ตายบน Windows
        batch=8,        # เดิม default 16 ลดเพื่อกัน VRAM เต็ม (ถ้ายัง OOM ใช้ 4)
        cache=False,    # ไม่เก็บภาพใน RAM

        # --- Data Augmentation ---
        degrees=15.0,
        shear=5.0,
        perspective=0.001,

        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10
    )