import json
import os

import numpy as np
import streamlit as st
from keras.applications import MobileNetV2
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from search_core import embed, preprocess_query, search

st.set_page_config(page_title="Doodle Search", page_icon="🎨")
HERE = os.path.dirname(__file__)


@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")


@st.cache_resource
def load_index():
    emb = np.load(os.path.join(HERE, "index", "embeddings.npy"))
    gallery = np.load(os.path.join(HERE, "index", "gallery_images.npy"))
    with open(os.path.join(HERE, "index", "config.json")) as f:
        cfg = json.load(f)
    return emb, gallery, cfg


model = load_model()
emb, gallery, cfg = load_index()

st.title("🎨 Doodle Search Engine")
st.write("Draw a doodle or upload a photo of a drawing. "
         "You'll see up to 25 doodles that look most like yours.")

mode = st.radio("Choose input method:", ["Draw", "Upload"], horizontal=True)

query_img = None
is_photo = False

if mode == "Draw":
    canvas = st_canvas(
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        update_streamlit=True,
        return_image_data=True,
        key="canvas",
    )
    if canvas.image_data is not None:
        query_img = Image.fromarray(canvas.image_data.astype("uint8"))

else:
    up = st.file_uploader("Upload your drawing", type=["jpg", "jpeg", "png"])
    if up is not None:
        query_img = ImageOps.exif_transpose(Image.open(up))
        is_photo = True

if query_img is not None:
    q = preprocess_query(query_img, photo=is_photo)

    if q is None:
        st.info("Nothing drawn yet. Draw something or add an image.")
    else:
        c1, c2 = st.columns(2)
        if is_photo:
            c1.image(query_img, caption="Your photo", width=200)
        c2.image(Image.fromarray(q).resize((200, 200), Image.NEAREST),
                 caption="What the model sees (28x28)", width=200)

        vec = embed(model, q)[0]
        idx, scores = search(vec, emb, cfg["top_k"])

        st.subheader("🔎 Similar Doodles")
        st.caption(f"Best match score: {scores[0]:.2f} (threshold {cfg['threshold']:.2f})")

        keep = scores >= cfg["threshold"]
        idx, scores = idx[keep], scores[keep]

        if len(idx) == 0:
            st.warning("Nothing similar found")
        else:
            st.caption(f"{len(idx)} doodles scored {cfg['threshold']:.2f} or above")
            for r in range(0, len(idx), 5):
                cols = st.columns(5)
                for c in range(min(5, len(idx) - r)):
                    i = r + c
                    cols[c].image(gallery[idx[i]].reshape(28, 28),
                                  caption=f"{scores[i]:.2f}",
                                  use_container_width=True)