import torch
import torch.utils
import torch.utils.checkpoint
import torch.utils.data
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import os
from PIL import Image

try:
    os.mkdir(r"cropped_dataset")
except Exception:
    print("cropped_dataset directory already exists")
try:
    os.mkdir(r"cropped_dataset/train")
except Exception:
    print("cropped_dataset/train directory already exists")
try:
    os.mkdir(r"cropped_dataset/test")
except Exception:
    print("cropped_dataset/test directory already exists")


mtcnn = MTCNN(image_size=244, margin=10, device="cuda", post_process=False)

no_faces = []
probs = []
for root, dir, file in os.walk(r"dataset/train"):
    for f in file:
        img = Image.open(os.path.join(root, f))
        img, prob = mtcnn(img, return_prob=True)
        probs.append(prob)
        if img is None:
            no_faces.append(img)
            print(os.path.join(root, f))
        else:
            try:
                os.mkdir(os.path.join(r"cropped_dataset/train", os.path.basename(root)))
            except Exception:
                pass
            cr_path = os.path.join(r"cropped_dataset/train", os.path.basename(root), f)
            plt.imsave(
                cr_path,
                img.to(torch.uint8).permute(1, 2, 0).numpy(),
            )
            print(f"saved image {cr_path}")
for i in no_faces:
    print(f"didn't find a face in {i} in training dataset")
print(f"# of files with no faces: {len(no_faces)} in training")
i = 0
for root, dir, file in os.walk(r"dataset/train"):
    for f in file:
        i += 1

print(f"there is {i} train photos")


no_faces = []
probs = []
for root, dir, file in os.walk(r"dataset/test"):
    for f in file:
        img = Image.open(os.path.join(root, f))
        img, prob = mtcnn(img, return_prob=True)
        probs.append(prob)
        if img is None:
            no_faces.append(img)
            print(os.path.join(root, f))
        else:
            try:
                os.mkdir(os.path.join(r"cropped_dataset/test", os.path.basename(root)))
            except Exception:
                pass
            cr_path = os.path.join(r"cropped_dataset/test", os.path.basename(root), f)
            plt.imsave(
                cr_path,
                img.to(torch.uint8).permute(1, 2, 0).numpy(),
            )
            print(f"saved image {cr_path}")
for i in no_faces:
    print(f"didn't find a face in {i} in testing dataset")
print(f"no of files with no faces: {len(no_faces)} in testing")
i = 0
for root, dir, file in os.walk(r"dataset/test"):
    for f in file:
        i += 1

print(f"there is {i} test photos")
