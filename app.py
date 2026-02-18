import streamlit as st
import torch
import torch.nn.functional as F
import timm
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

st.set_page_config(page_title="Oral Cancer Detection", page_icon="🦷")

# ==================================================
# LOAD IMAGE MODEL
# ==================================================
@st.cache_resource
def load_image_model():
    model = timm.create_model("eva02_base_patch14_224", pretrained=False, num_classes=2)
    ckpt = torch.load("eva02_oralcancer_best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

image_model = load_image_model()

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(img):
    x = val_transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(image_model(x), dim=1)[0]
    return float(probs[1])  # CLASS-1

# ==================================================
# LOAD METADATA MODEL
# ==================================================
@st.cache_resource
def load_meta():
    model = pickle.load(open("metadata_lgbm_model_4feat.pth","rb"))
    scaler = pickle.load(open("metadata_scaler.pth","rb"))
    return model, scaler

meta_model, scaler = load_meta()

def predict_meta(meta):
    df = pd.DataFrame([meta])
    df["age"] = scaler.transform(df[["age"]])
    return float(meta_model.predict_proba(df)[0][1])  # CLASS-1

# ==================================================
# UI
# ==================================================
st.title("🦷 Oral Cancer Detection ")

uploaded_img = st.file_uploader("📸 Upload Oral Image", type=["jpg","jpeg","png"])

st.subheader("Patient Metadata")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male","Female"])
with c2:
    alcohol = st.selectbox("Alcohol", ["Yes","No"])
    chewing = st.selectbox("Betel Quid Chewing", ["Yes","No"])

# ==================================================
# PREDICTION
# ==================================================
if uploaded_img:

    img = Image.open(uploaded_img)
    st.image(img, width=280)

    # ---------- IMAGE ----------
    image_prob = predict_image(img)
    st.subheader("🖼️ Image Model Prediction")
    st.write(f"**Oral Cancer Probability:** `{image_prob:.4f}`")

    # ---------- METADATA ----------
    meta_prob = predict_meta({
        "age": age,
        "gender": 1 if gender=="Male" else 0,
        "alcohol": 1 if alcohol=="Yes" else 0,
        "chewing_betel_quid": 1 if chewing=="Yes" else 0
    })

    st.subheader("📋 Metadata Model Prediction")
    st.write(f"**Oral Cancer Probability:** `{meta_prob:.4f}`")

    # ---------- FINAL ----------
    st.divider()
    st.subheader("🧠 Final Late Fusion Prediction")

    final_prob = (0.7 * image_prob) + (0.3 * meta_prob)

    st.markdown(
        f"""
        <h1 style="text-align:center;color:#b30000;">
        {final_prob:.4f}
        </h1>
        """,
        unsafe_allow_html=True
    )

    if final_prob >= 0.5:
        st.error("🚨 ORAL CANCER DETECTED")
    else:
        st.success("✅ HEALTHY")
