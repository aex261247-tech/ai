# app.py
# Streamlit app: อัปโหลดรูป -> YOLOv8 (seg) แยกชิ้นเสื้อผ้า -> ดูดสีหลัก -> แนะนำสี Complementary/Analogous/Triadic
# ใช้โมเดล best.pt ที่เทรนแบบ "segment" แล้ว (วางไฟล์ไว้โฟลเดอร์เดียวกับ app.py)

import io
import colorsys
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.cluster import KMeans
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="AI Stylist – Color Recommender", layout="centered")

MODEL_PATH = "best.pt"  # เปลี่ยนได้จาก sidebar
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

# ชื่อคลาส EN -> TH
EN2TH = {
    "long_sleeved_dress": "ชุดเดรส",
    "long_sleeved_outwear": "เสื้อคลุมแขนยาว",
    "long_sleeved_shirt": "เสื้อแขนยาว",
    "short_sleeved_dress": "ชุดเดรส",
    "short_sleeved_outwear": "เสื้อคลุมแขนสั้น",
    "short_sleeved_shirt": "เสื้อแขนสั้น",
    "shorts": "กางเกงขาสั้น",
    "skirt": "กระโปรง",
    "sling": "เสื้อสายเดี่ยว",
    "sling_dress": "ชุดเดรส",
    "trousers": "กางเกงขายาว",
    "vest": "เสื้อกั๊ก",
    "vest_dress": "ชุดเดรส",
}

# ----------------------------
# Utils: สี
# ----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def hsv01_to_rgb(h, s, v):
    """รับ h,s,v ในช่วง 0..1 -> คืน (r,g,b) 0..255 และ hex"""
    r, g, b = colorsys.hsv_to_rgb(clamp01(h), clamp01(s), clamp01(v))
    R = int(round(r * 255))
    G = int(round(g * 255))
    B = int(round(b * 255))
    return (R, G, B), f"#{R:02X}{G:02X}{B:02X}"

def rgb255_to_hsv01(rgb):
    r, g, b = rgb
    return colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)  # h,s,v ∈ [0,1]

def color_harmonies_from_rgb(rgb):
    """คำนวณ Complementary / Analogous(±30°) / Triadic(±120°) จาก RGB"""
    h, s, v = rgb255_to_hsv01(rgb)

    # Complementary = 180°
    comp_rgb, comp_hex = hsv01_to_rgb(h + 0.5, s, v)

    # Analogous ≈ ±30° -> ±0.0833
    ana1_rgb, ana1_hex = hsv01_to_rgb(h + 0.0833, s, v)
    ana2_rgb, ana2_hex = hsv01_to_rgb(h - 0.0833, s, v)

    # Triadic = ±120° -> ±1/3
    tri1_rgb, tri1_hex = hsv01_to_rgb(h + 1/3, s, v)
    tri2_rgb, tri2_hex = hsv01_to_rgb(h - 1/3, s, v)

    return {
        "base": (rgb, f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"),
        "complementary": (comp_rgb, comp_hex),
        "analogous": [(ana1_rgb, ana1_hex), (ana2_rgb, ana2_hex)],
        "triadic": [(tri1_rgb, tri1_hex), (tri2_rgb, tri2_hex)],
    }

def hex_chip(hex_str: str, label: str = ""):
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
          <div style="width:28px;height:28px;border-radius:6px;border:1px solid #ddd;background:{hex_str};"></div>
          <div style="font-size:0.95rem;">{label} <code style="opacity:0.8">{hex_str}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Utils: dominant color
# ----------------------------
def filter_pixels_rgb(pixels: np.ndarray) -> np.ndarray:
    """กรองพิกเซลที่ซีด/ขาว/ดำ/เทาออกคร่าว ๆ เพื่อลดสัญญาณรบกวน"""
    if pixels.size == 0:
        return pixels
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    # ตัดขาวจัดและดำจัด
    not_white = ~((R > 235) & (G > 235) & (B > 235))
    not_black = ~((R < 15) & (G < 15) & (B < 15))
    # ตัดเทา (ความต่าง R,G,B ใกล้กันมาก)
    not_gray = ~((np.abs(R - G) < 12) & (np.abs(G - B) < 12))
    keep = not_white & not_black & not_gray
    return pixels[keep]

def dominant_color_from_masked_pixels(pixels: np.ndarray, n_clusters: int = 3):
    """
    pixels: (N,3) uint8 (RGB) – return (rgb_tuple, hex)
    """
    if pixels.size == 0:
        return (200, 200, 200), "#C8C8C8"

    px = filter_pixels_rgb(pixels)
    if px.size == 0:
        px = pixels

    # subsample เพื่อความเร็ว
    if len(px) > 6000:
        idx = np.random.choice(len(px), 6000, replace=False)
        px = px[idx]

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    labels = km.fit_predict(px.astype(np.float32))
    centers = km.cluster_centers_.astype(int)
    # เลือกคลัสเตอร์ที่เยอะสุด
    _, counts = np.unique(labels, return_counts=True)
    dom_idx = int(np.argmax(counts))
    rgb = tuple(int(x) for x in centers[dom_idx])
    hex_str = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    return rgb, hex_str

# ----------------------------
# Load model (cache)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    m = YOLO(path)
    return m

# ----------------------------
# UI
# ----------------------------
st.title("🧥 AI Stylist – แยกเสื้อผ้า ดูดสี แนะนำคู่สี")
with st.sidebar:
    st.header("ตั้งค่า")
    model_path = st.text_input("ตำแหน่งโมเดล (.pt)", MODEL_PATH)
    conf = st.slider("Confidence", 0.05, 0.85, DEFAULT_CONF, 0.01)
    iou = st.slider("IoU", 0.2, 0.9, DEFAULT_IOU, 0.01)
    imgsz = st.select_slider("Image size (สั้นสุดพอ)", options=[320, 480, 640, 800], value=DEFAULT_IMGSZ)
    st.caption("วางไฟล์ best.pt ไว้โฟลเดอร์เดียวกับ app.py แล้วปรับพาธได้ตามต้องการ")

# โหลดโมเดล
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

if getattr(model, "task", None) != "segment":
    st.warning("⚠️ โมเดลนี้ดูเหมือนจะไม่ใช่ **segmentation** (ต้องเป็น YOLOv8-seg) — ผลลัพธ์อาจไม่มี masks")
st.caption(f"คลาสในโมเดล: {model.names}")

uploaded = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png", "webp"])
if not uploaded:
    st.info("อัปโหลดรูปแล้วระบบจะวิเคราะห์ให้เลย ✨")
    st.stop()

# อ่านภาพ (RGB)
img_pil = Image.open(uploaded).convert("RGB")
img_np = np.array(img_pil)

with st.spinner("กำลังตรวจจับ…"):
    res_list = model.predict(img_np, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
res = res_list[0]

# รูป anotated
annot_bgr = res.plot()  # BGR
annot_rgb = annot_bgr[:, :, ::-1]  # to RGB
st.image([img_pil, annot_rgb], caption=["ภาพต้นฉบับ", "ผลลัพธ์การตรวจจับ"], use_column_width=True)

if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
    st.warning("ไม่พบชิ้นเสื้อผ้า (masks) จากภาพนี้")
    st.stop()

# เตรียมข้อมูล
masks = res.masks.data.cpu().numpy()            # (N, h, w) float 0..1
H, W = img_np.shape[:2]
boxes = res.boxes
names = model.names

st.subheader("ผลลัพธ์รายชิ้น")
for i in range(len(masks)):
    # class / conf
    cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
    conf_i = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
    name_en = names.get(cls_id, f"id:{cls_id}")
    name_th = EN2TH.get(name_en, name_en)

    # mask -> resize เป็นขนาดภาพ
    m_small = masks[i]  # 0..1
    m_img = Image.fromarray((m_small > 0.5).astype(np.uint8) * 255)
    m_big = m_img.resize((W, H), Image.NEAREST)
    mask_bool = np.array(m_big).astype(bool)

    # pixels ภายใน mask (RGB)
    pixels = img_np[mask_bool]

    # dominant color + harmonies
    dom_rgb, dom_hex = dominant_color_from_masked_pixels(pixels, n_clusters=3)
    har = color_harmonies_from_rgb(dom_rgb)

    with st.container(border=True):
        st.markdown(f"**{name_th}**  ·  *{name_en}*  ·  conf={conf_i:.2f}")
        hex_chip(har["base"][1], "สีหลัก")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Complementary**")
            hex_chip(har["complementary"][1])
        with col2:
            st.markdown("**Analogous**")
            for _, hx in har["analogous"]:
                hex_chip(hx)
        with col3:
            st.markdown("**Triadic**")
            for _, hx in har["triadic"]:
                hex_chip(hx)

# ปุ่มดาวน์โหลดรูป anotated
buf = io.BytesIO()
Image.fromarray(annot_rgb).save(buf, format="PNG")
st.download_button("⬇️ ดาวน์โหลดภาพผลลัพธ์ (PNG)", data=buf.getvalue(), file_name="result.png", mime="image/png")

st.markdown('''<div class="footer">
<span style="font-size:1em;font-weight:600;color:#404040;">พัฒนาโดย Chanaphon Phetnoi</span><br>
<span style="font-size:0.95em;color:#404040;">รหัสนักศึกษา 664230017 | ห้อง 66/45</span><br>
<span style="font-size:0.95em;color:#404040;">นักศึกษาสาขาเทคโนโลยีสารสนเทศ</span>
</div>''', unsafe_allow_html=True)
