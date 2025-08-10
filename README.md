# AI Stylist

AI Stylist เป็นแอป **Streamlit** สำหรับวิเคราะห์โทนสีและสไตล์แฟชั่นจากภาพถ่าย พร้อมแนะนำโทนสีที่เข้ากันโดยอัตโนมัติ

---

## คุณสมบัติหลัก

- วิเคราะห์ **dominant color** ของเสื้อและกางเกง
- แสดง **HEX / RGB / ชื่อโทนภาษาไทย** และพาเลตสีที่เข้ากัน
- แนะนำสีตามทฤษฎี **Complementary / Analogous / Triadic** (คำนวณบน HSV)
- สรุป **แนวสไตล์โดยรวม (Overall Style)**
- ลบพื้นหลังอัตโนมัติ (รองรับหลายวิธี)
- UI สวยงาม รองรับภาษาไทย
- (ตัวเลือก) ใช้โมเดล **YOLOv8-SEG** (`best.pt`) เพื่อจับ “ชนิดชุด” เช่น เสื้อ+กางเกง / เสื้อ+กระโปรง / เดรส ฯลฯ

---

## โครงสร้างโปรเจกต์

```
ai-stylist/
├─ app.py                  # โค้ดหลักของแอป Streamlit
├─ color_table_th.py       # ตารางชื่อสี/โทนภาษาไทย
├─ segmentation_utils.py   # ฟังก์ชันแยกส่วนเสื้อ/กางเกง
├─ requirements.txt        # รายการไลบรารี
├─ best.pt                 # (ตัวเลือก) โมเดล YOLOv8-SEG ที่เทรนแล้ว
└─ README.md
```

> ถ้าใช้งาน YOLO ตรวจจับชนิดชุด ให้ใส่ไฟล์โมเดล `best.pt` ไว้ข้าง `app.py` และปรับ `MODEL_PATH` ในโค้ดให้ตรง

---

## วิธีติดตั้งและรัน (แนะนำ Python 3.11)

### Windows (PowerShell)

```powershell
# 1) สร้างและเปิดใช้งาน virtual env
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) อัปเกรดตัวติดตั้งพื้นฐาน
python -m pip install -U pip setuptools wheel

# 3) ติดตั้งไลบรารีของโปรเจกต์
pip install -r requirements.txt

# (ทางเลือก) ถ้าใช้ YOLO + ไม่มี PyTorch อยู่ ให้ติดตั้งล้อ CPU ทางการก่อน
# pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# 4) รันแอป
streamlit run app.py
```

### macOS / Linux (bash)

```bash
# 1) สร้างและเปิดใช้งาน virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2) อัปเกรดตัวติดตั้งพื้นฐาน
python -m pip install -U pip setuptools wheel

# 3) ติดตั้งไลบรารีของโปรเจกต์
pip install -r requirements.txt

# (ทางเลือก) ติดตั้ง PyTorch CPU ถ้าจะใช้ YOLO
# pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# 4) รันแอป
streamlit run app.py
```

เปิดลิงก์ที่แสดง (Local URL: http://localhost:8501) จากนั้นอัปโหลดรูปภาพเพื่อดูผลลัพธ์

---

## วิธีใช้งาน

1. เปิดเว็บแอป http://localhost:8501
2. กด **Upload image** แล้วเลือกรูปบุคคล
3. ระบบจะแยกส่วนเสื้อผ้า → ดูดค่าสีหลัก → แนะนำพาเลต Complementary / Analogous / Triadic → แสดงผลพร้อม HEX/RGB/ชื่อโทน
4. (ถ้าต่อ YOLO) จะแสดงประเภทชุด/ชิ้นเสื้อผ้าที่ตรวจจับได้ด้วย

---

## ไลบรารีที่ใช้

- streamlit
- numpy
- pillow
- scikit-learn
- webcolors
- rembg
- backgroundremover
- opencv-python (หรือ opencv-python-headless สำหรับเซิร์ฟเวอร์)
- requests
- (ตัวเลือก) ultralytics + torch + torchvision สำหรับ YOLOv8-SEG

ติดตั้งทั้งหมดได้ด้วย

```bash
pip install -r requirements.txt
```

---

## ทดสอบโมเดล YOLO (ถ้าใช้งาน)

วาง `best.pt` ไว้ข้าง `app.py` แล้วทดสอบว่าโหลดได้:

```python
from ultralytics import YOLO
m = YOLO("best.pt")
print("task:", m.task)      # ควรเป็น 'segment'
print("classes:", m.names)  # รายชื่อคลาส
```

---

## Mapping EN → TH ที่ใช้บ่อย

| EN                   | TH                      |
|----------------------|-------------------------|
| long_sleeved_dress   | เดรสแขนยาว              |
| long_sleeved_outwear | เสื้อคลุมแขนยาว            |
| long_sleeved_shirt   | เชิ้ตแขนยาว               |
| short_sleeved_dress  | เดรสแขนสั้น               |
| short_sleeved_outwear| เสื้อคลุมแขนสั้น             |
| short_sleeved_shirt  | เสื้อยืด/เชิ้ตแขนสั้น          |
| shorts               | กางเกงขาสั้น               |
| skirt                | กระโปรง                  |
| sling                | สายเดี่ยว (ท่อนบน)         |
| sling_dress          | เดรสสายเดี่ยว              |
| trousers             | กางเกงขายาว              |
| vest                 | เสื้อกั๊ก                   |
| vest_dress           | เดรสแขนกุด/สไตล์เสื้อกั๊ก     |

---

## เคล็ดลับ / แก้ปัญหา

- แนะนำ Python 3.11 เพื่อหลีกเลี่ยงการคอมไพล์แพ็กเกจบน Windows
- ถ้า `ModuleNotFoundError: ultralytics` → `pip install ultralytics`
- ถ้า `cv2` ไม่เจอ → `pip install opencv-python-headless==4.9.0.80`
- Port 8501 ชน → `streamlit run app.py --server.port 8502`
- หากลบพื้นหลังช้า/ล้มเหลว ลองใช้ภาพสว่างและพื้นหลังเรียบ หรือปิดขั้นตอนที่ไม่จำเป็นในโค้ด

---

## เครดิต

พัฒนาโดย Chanaphon Phetnoi (รหัสนักศึกษา 664230017)

โค้ดนี้ใช้เพื่อการศึกษาและสาธิตเท่านั้น
streamlit run app.py
macOS / Linux (bash)

bash
คัดลอก
แก้ไข
# 1) สร้างและเปิดใช้งาน virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2) อัปเกรดตัวติดตั้งพื้นฐาน
python -m pip install -U pip setuptools wheel

# 3) ติดตั้งไลบรารีของโปรเจกต์
pip install -r requirements.txt

# (ทางเลือก) ติดตั้ง PyTorch CPU ถ้าจะใช้ YOLO
# pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# 4) รันแอป
streamlit run app.py
เปิดลิงก์ที่แสดง (Local URL: http://localhost:8501) จากนั้นอัปโหลดรูปภาพเพื่อดูผลลัพธ์

วิธีใช้งาน
เปิดเว็บแอป http://localhost:8501

กด Upload image แล้วเลือกรูปบุคคล

ระบบจะแยกส่วนเสื้อผ้า → ดูดค่าสีหลัก → แนะนำพาเลต Complementary / Analogous / Triadic → แสดงผลพร้อม HEX/RGB/ชื่อโทน

(ถ้าต่อ YOLO) จะแสดงประเภทชุด/ชิ้นเสื้อผ้าที่ตรวจจับได้ด้วย

ไลบรารีที่ใช้
streamlit

numpy

pillow

scikit-learn

webcolors

rembg

backgroundremover

opencv-python (หรือ opencv-python-headless สำหรับเซิร์ฟเวอร์)

requests

(ตัวเลือก) ultralytics + torch + torchvision สำหรับ YOLOv8-SEG

ติดตั้งทั้งหมดได้ด้วย pip install -r requirements.txt

ทดสอบโมเดล YOLO (ถ้าใช้งาน)
วาง best.pt ไว้ข้าง app.py แล้วทดสอบว่าโหลดได้:

python
คัดลอก
แก้ไข
from ultralytics import YOLO
m = YOLO("best.pt")
print("task:", m.task)      # ควรเป็น 'segment'
print("classes:", m.names)  # รายชื่อคลาส
Mapping EN → TH ที่ใช้บ่อย

long_sleeved_dress → เดรสแขนยาว

long_sleeved_outwear → เสื้อคลุมแขนยาว

long_sleeved_shirt → เชิ้ตแขนยาว

short_sleeved_dress → เดรสแขนสั้น

short_sleeved_outwear → เสื้อคลุมแขนสั้น

short_sleeved_shirt → เสื้อยืด/เชิ้ตแขนสั้น

shorts → กางเกงขาสั้น

skirt → กระโปรง

sling → สายเดี่ยว (ท่อนบน)

sling_dress → เดรสสายเดี่ยว

trousers → กางเกงขายาว

vest → เสื้อกั๊ก

vest_dress → เดรสแขนกุด/สไตล์เสื้อกั๊ก

เคล็ดลับ / แก้ปัญหา
แนะนำ Python 3.11 เพื่อหลีกเลี่ยงการคอมไพล์แพ็กเกจบน Windows

ถ้า ModuleNotFoundError: ultralytics → pip install ultralytics

ถ้า cv2 ไม่เจอ → pip install opencv-python-headless==4.9.0.80

Port 8501 ชน → streamlit run app.py --server.port 8502

หากลบพื้นหลังช้า/ล้มเหลว ลองใช้ภาพสว่างและพื้นหลังเรียบ หรือปิดขั้นตอนที่ไม่จำเป็นในโค้ด

เครดิต
พัฒนาโดย Chanaphon Phetnoi (รหัสนักศึกษา 664230017)

โค้ดนี้ใช้เพื่อการศึกษาและสาธิตเท่านั้น