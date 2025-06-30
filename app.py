import streamlit as st
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import sys
import json
import matplotlib.pyplot as plt


import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"


from doctane.ocr_pipeline.ocr_predictor_dp import OCRPredictor
from doctane.models.detection.smp_model_zoo import (
    seg_linknet_resnet50,
    seg_deeplabv3plus_resnet50,
    seg_segformer_resnet50
)
from doctane.models.recognition.models import sar_resnet34, sar_resnet18


# Function to store the document (type-object) 
def convert_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj) 

# -- Define available models --
dict_det = {
    "Segformer-ResNet50": {
    "loader": seg_segformer_resnet50,
    "ckpt": ""
    },
    "LinkNet-ResNet50": {
        "loader": seg_linknet_resnet50,
        "ckpt": ""
    },
    "DeepLabV3Plus-ResNet50": {
        "loader": seg_deeplabv3plus_resnet50,
        "ckpt": ""
    },
}

dict_recog = {
    "SAR-ResNet34": {
        "loader": sar_resnet34,
        "ckpt": ""
    },
    "SAR-ResNet18": {
        "loader": sar_resnet18, 
        "ckpt": ""
    },
}

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>🧾 DOCTANE </h1>",
    unsafe_allow_html=True
)

# --- Define columns layout ---
col_left, col_mid, col_right = st.columns([1, 2, 2])

# --- Column: LEFT - Model & Image Upload ---
with col_left:
    st.header("⚙️ Choose the Model...")

    selected_det = st.selectbox("Detection Model", list(dict_det.keys()))
    selected_recog = st.selectbox("Recognition Model", list(dict_recog.keys()))

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Proceed if image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(image)

    # Load selected detection model
    det_loader = dict_det[selected_det]["loader"]
    det_ckpt = dict_det[selected_det]["ckpt"]
    det_model = det_loader(pretrained=False)
    det_model.load_state_dict(torch.load(det_ckpt, map_location='cpu'))

    # Load selected recognition model
    recog_loader = dict_recog[selected_recog]["loader"]
    recog_ckpt = dict_recog[selected_recog]["ckpt"]
    recog_model = recog_loader(pretrained=False, pretrained_backbone=False)
    recog_model.load_state_dict(torch.load(recog_ckpt, map_location='cpu'), strict=False)

    # Create OCR predictor
    model = OCRPredictor(det_predictor=det_model, reco_predictor=recog_model)

    with st.spinner("🔍 Running OCR..."):
        output = model([np_image])

    # --- Column: MIDDLE - Visualization ---
    with col_mid:
        st.header("📊 Visualization")
        st.markdown(f"`Detection:` **{selected_det}** | `Recognition:` **{selected_recog}**")
        fig = plt.figure()
        output.show()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        st.image(buf, caption="OCR Output", use_container_width=True)

    # --- Column: RIGHT - JSON Output ---  
    with col_right:
        st.header("📦 OCR Output")
        try:
            if hasattr(output, 'pages'):
                json_data = [page.__dict__ for page in output.pages]
                st.json(json_data)

                # Download button
                json_str = json.dumps(json_data, indent=2, default=convert_to_dict)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name="ocr_output.json",
                    mime="application/json"
                )
            else:
                st.text(str(output))
        except Exception as e:
            st.error(f"Failed to parse OCR output: {e}")
else:
    with col_mid:
        st.info("📷 Please upload an image to begin.")
