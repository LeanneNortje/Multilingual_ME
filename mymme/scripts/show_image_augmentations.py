import random
import streamlit as st

from PIL import Image
from torchvision import transforms as T

from mymme.data import MEDataset, get_image_path, transform_image_train_1


dataset = MEDataset("train", ("english",))
data = dataset.image_files


def show1(image, num_samples=5):
    st.image(image)
    cols = st.columns(num_samples)
    for i in range(num_samples):
        cols[i].image(transform_image_train_1(image))
    st.markdown("---")


idxs = random.sample(range(len(data)), 10)
for i in idxs:
    datum = data[i]
    path = get_image_path(datum)
    image = Image.open(path)
    show1(image)
