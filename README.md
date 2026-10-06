# YOLO26 - Board Game Detection (Exploding Kittens, Insider & UNO)

โปรเจกต์ Object Detection สำหรับตรวจจับกล่องบอร์ดเกม 3 ประเภท ได้แก่ **Exploding Kittens (HALLSExplodingKit)**, **Insider** และ **UNO** โดยใช้ YOLO26

## Project Overview

วัตถุประสงค์คือพัฒนาโมเดล Computer Vision สำหรับตรวจจับกล่องบอร์ดเกมจากภาพ โดยแบ่งวัตถุออกเป็น 3 Class

| ID | Class |
|----|-------|
| 0 | HALLSExplodingKit |
| 1 | Insider |
| 2 | UNO |

## Features

- ตรวจจับกล่องบอร์ดเกม 3 ประเภทด้วย Bounding Box
- ตรวจจับจากรูปภาพ
- ตรวจจับ Real-time ผ่าน Webcam
- Train และ Validate ด้วย Ultralytics YOLO
- มีโมเดลที่ Train แล้ว (`best.pt`) พร้อมใช้งาน
- มีชุดข้อมูลตัวอย่าง (`sample_dataset/`) สำหรับลองเทรน

## Technology Stack

| Technology | ใช้สำหรับ |
|------------|-----------|
| Python | พัฒนาโปรแกรม |
| YOLO26 (nano) | Object Detection |
| Ultralytics | Training และ Inference |
| OpenCV | ประมวลผลภาพและ Webcam |
| PyTorch | Deep Learning |
| Label Studio | ทำ Annotation / Bounding Box |

## Project Structure

```
AI_YOLO_Board_Game/
│
├── sample_dataset/          # ข้อมูลตัวอย่าง (15 train / 8 val)
│   ├── data.yaml
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── 01-export_dataset.py     # เตรียมและแปลง Dataset
├── 02-train.py              # Train โมเดล
├── 03-test_image.py         # ทดสอบกับรูปภาพ
├── 05-test-camera.py        # ตรวจจับผ่าน Webcam
├── best.pt                  # โมเดลที่ Train แล้ว
├── requirements.txt         # Python packages
├── .gitignore
└── README.md
```

| File / Folder | Description |
|---------------|-------------|
| `best.pt` | โมเดล YOLO ที่ Train แล้ว ใช้สำหรับตรวจจับ |
| `sample_dataset/` | ชุดข้อมูลตัวอย่างขนาดเล็ก (~1 MB) |
| `01-export_dataset.py` | เตรียมและแปลง Dataset |
| `02-train.py` | Train โมเดล |
| `03-test_image.py` | ทดสอบโมเดลกับรูปภาพ |
| `05-test-camera.py` | ตรวจจับวัตถุผ่าน Webcam แบบ Real-time |
| `requirements.txt` | รายการ Python packages |

## Project Workflow

```
 Prepare Image → Annotation (Label Studio)
        ↓
 01-export_dataset.py
        ↓
     data.yaml
        ↓
   02-train.py
        ↓
     best.pt
        │
   ┌────┴─────────────┐
   ↓                  ↓
03-test_image.py   05-test-camera.py
 (Image Test)       (Webcam Test)
```

## Installation

สร้าง virtual environment

```bash
python -m venv env
```

เปิดใช้งาน (PowerShell)

```powershell
.\env\Scripts\Activate.ps1
```

ติดตั้ง package

```bash
pip install -r requirements.txt
```

> **หมายเหตุ**
> - ถ้าใช้ GPU ให้ติดตั้ง PyTorch เวอร์ชัน CUDA ตามคู่มือที่ https://pytorch.org
> - ถ้า `cv2.imshow` ขึ้น error เรื่อง GUI ให้ใช้ `opencv-python` ตัวเดียว ห้ามติดตั้งร่วมกับ `opencv-python-headless`

## Dataset

| ชุดข้อมูล | รายละเอียด |
|-----------|------------|
| ตัวอย่าง | โฟลเดอร์ `sample_dataset/` ใน repo นี้ |
| ข้อมูลเต็ม (~827 รูป, 3 Class) | [ดาวน์โหลดจาก Google Drive](ใส่ลิงก์ที่นี่) |

วิธีใช้ข้อมูลเต็ม: แตกไฟล์ `dataset.zip` ไว้ในโฟลเดอร์โปรเจกต์ แล้วแก้ `path` ใน `data.yaml` ให้ตรงกับเครื่องของคุณ

ตัวอย่างโครงสร้าง:

```
dataset/
├── images/{train,val}/
├── labels/{train,val}/
└── data.yaml
```

## Training

```bash
python 02-train.py
```

หรือลองกับข้อมูลตัวอย่าง:

```bash
yolo detect train data="sample_dataset/data.yaml" model=yolo26n.pt epochs=10 imgsz=640 batch=8 workers=0
```

ค่าที่ใช้เทรนโมเดลหลัก: `epochs=150`, `imgsz=640`, `optimizer=MuSGD`, พร้อม Data Augmentation (`degrees`, `shear`, `perspective`, `fliplr`, `mosaic`, `mixup`)

> **Windows:** ถ้าเจอ `DataLoader worker exited unexpectedly` หรือ CUDA OOM ให้ตั้ง `workers=0` และลด `batch` (เช่น 8 หรือ 4)

## Validation

```bash
yolo detect val model="best.pt" data="sample_dataset/data.yaml" imgsz=640
```

YOLO จะแสดง Precision, Recall, mAP50 และ mAP50-95 ทั้งแบบรวมและแยก Class

## Image Detection

```bash
# ระบุรูปเอง
python 03-test_image.py path/to/image.jpg

# ไม่ระบุ จะใช้รูปจาก sample_dataset/images/val
python 03-test_image.py
```

ผลลัพธ์จะถูกบันทึกในโฟลเดอร์ `runs/detect/predict...`

## Real-time Webcam Detection

```bash
python 05-test-camera.py
```

```
Webcam → OpenCV → YOLO Model → Bounding Box + Class
```

ค่าปัจจุบัน: `conf = 0.5`, `camera = 0` (ใช้ GPU ถ้ามี ไม่งั้นใช้ CPU) กด `q` เพื่อออก

## Training Results

ผลจากชุด Validation (168 รูป) ด้วย `best.pt`

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **All** | 0.989 | 0.939 | **0.953** | **0.718** |
| HALLSExplodingKit | 0.987 | 0.865 | 0.902 | 0.714 |
| Insider | 0.984 | 0.980 | 0.983 | 0.673 |
| UNO | 0.996 | 0.972 | 0.975 | 0.767 |

- เทรน 150 epochs ใช้เวลาประมาณ 3 ชั่วโมง (RTX 3050 6GB Laptop)
- Inference ~1.7 ms/ภาพ ขนาดไฟล์โมเดล ~5.4 MB

## Dataset Recommendations

เพื่อเพิ่มความแม่นยำ ควรมีภาพที่หลากหลาย เช่น

- มุมกล้องหลายมุม
- ระยะใกล้และระยะไกล
- พื้นหลังหลายรูปแบบ
- ระดับแสงแตกต่างกัน
- วัตถุหลายชิ้นในภาพเดียว
- ภาพจาก Webcam หรือสภาพแวดล้อมที่ใกล้เคียงการใช้งานจริง

โดยเฉพาะ **HALLSExplodingKit** ซึ่งมี Recall ต่ำสุด (0.865) ควรเพิ่มภาพในมุมและแสงที่หลากหลายขึ้น

## Notes

- ใช้ `best.pt` สำหรับ Inference
- ควรแยก Test set ออกจาก Train / Validation หากต้องการวัดผลกับรูปที่โมเดลไม่เคยเห็นจริง
- ตรวจสอบ Label ให้พิกัดอยู่ในช่วง 0-1 ก่อนเทรน ไม่เช่นนั้นรูปนั้นจะถูกข้าม
- ชุด Validation มีขนาดเล็ก ผลจริงอาจต่างจากตัวเลขเล็กน้อย

## Project Summary

โปรเจกต์นี้เป็นระบบ Object Detection ด้วย YOLO26 สำหรับตรวจจับกล่องบอร์ดเกม ได้แก่ Exploding Kittens, Insider และ UNO ครอบคลุมตั้งแต่การเตรียม Dataset, Training, ทดสอบกับรูปภาพ และนำโมเดลไปใช้กับ Webcam แบบ Real-time

```
Dataset → Training → best.pt ─┬─► Image Detection (03-test_image.py)
                              └─► Real-time Detection (05-test-camera.py)
```

**Project Status:** Completed / Ready for Testing
