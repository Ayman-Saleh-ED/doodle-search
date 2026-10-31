import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
from keras.applications.mobilenet_v2 import preprocess_input


def to_model_input(images_28, invert=False):
    x = np.asarray(images_28, dtype="float32").reshape(-1, 28, 28, 1)
    if invert:
        x = 255.0 - x
    x = tf.image.resize(x, (224, 224)).numpy()
    x = np.repeat(x, 3, axis=-1)
    return preprocess_input(x)


def embed(model, images_28, invert=False, batch_size=64):
    out = []
    for i in range(0, len(images_28), batch_size):
        out.append(model.predict(to_model_input(images_28[i:i + batch_size], invert), verbose=0))
    emb = np.concatenate(out)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def search(query_vector, embeddings, k):
    scores = embeddings @ query_vector
    idx = np.argsort(scores)[::-1][:k]
    return idx, scores[idx]


def preprocess_query(image, photo=False, thr=140):
    img = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA")).convert("L")
    img = ImageOps.invert(img)                        # strokes white on black
    if photo:                                         # photos: remove paper shading
        img = ImageOps.autocontrast(img)
        img = img.point(lambda p: 255 if p > thr else 0)
    box = img.getbbox()
    if box is None:
        return None                                   # blank
    img = ImageOps.pad(img.crop(box), (20, 20))
    if photo:
        img = img.filter(ImageFilter.MaxFilter(3))    # thicken thin pen lines
    return np.array(ImageOps.expand(img, border=4, fill=0))