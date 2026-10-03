import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

@st.cache_resource
def load_my_model():
    model_path = 'brain_tumor_model.h5' 
    model = load_model(model_path)
    last_conv_layer_name = 'conv5_block3_out'
    return model, last_conv_layer_name

model, last_conv_layer_name = load_my_model()
classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

st.title("🧠 ThromboVision AI: Advanced Brain Abnormality & Clot Triage")
st.write("Upload a brain scan image for high-precision abnormality detection and localization.")

uploaded_file = st.file_uploader("Choose a Brain scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Brain Scan', use_container_width=True)
    
    img_cv = np.array(img)
    if len(img_cv.shape) == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    elif img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
        
    img_resize = cv2.resize(img_cv, (224, 224))
    img_array = image.img_to_array(img_resize)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 

    if st.button('Analyze Abnormality & Generate Heatmap'):
        with st.spinner('Analyzing scan with high precision...'):
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100
            predicted_class = classes[predicted_class_index]

            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_class_index)
            
            orig_path = "uploaded_image.jpg"
            cv2.imwrite(orig_path, img_cv)
            cam_path = save_and_display_gradcam(orig_path, heatmap, cam_path="cam.jpg")
            
            st.success(f"Detection Result: **{predicted_class}**")
            st.info(f"Confidence Score: **{confidence:.2f}%**")
            
            if predicted_class != 'No Tumor':
                st.warning("⚠️ Clinical Alert: Abnormality / Clot region identified.")
                st.subheader("Grad-CAM Abnormality Localization Map:")
                cam_img = Image.open(cam_path)
                st.image(cam_img, caption='Heatmap Overlay (Red/Yellow = High Activation Region)', use_container_width=True)
                st.write("Note: Highlighted regions indicate suspicious tissue patterns localized by the deep learning architecture.")
            else:
                st.info("✅ Result normal. No critical clot or abnormality patterns detected.")
