from pathlib import Path
from matplotlib import cm
from sklearn.preprocessing import minmax_scale
import numpy as np
from PIL import Image

def create_dir(path):
    Path(path).mkdir(exist_ok=True)

def one_to_one_scaler(feature):
    return minmax_scale(feature.ravel(), feature_range=(-1, 1)).reshape(feature.shape)

def rgb_scaler(feature):
    return np.uint8(minmax_scale(feature.ravel(), feature_range=(0, 255)).reshape(feature.shape))

def rgb_lookup_table_converter(feature):
    colormap = cm.get_cmap("rainbow")
    lut = (np.array([colormap(i/255)[:3] for i in range(256)]) * 255)

    return np.array(np.take(lut, feature, axis=0), dtype=np.uint8)

def label_converter(lookup, labels):
    return [lookup[label] for label in labels]

# def save_image(X, save_path, channel=3, pixel_size=512):
#     if channel == 3:
#         image = Image.fromarray(X, "RGB")
#     else:
#         image = Image.fromarray(X, "L")
#     image = image.resize((pixel_size, pixel_size))
#     image.save(save_path)