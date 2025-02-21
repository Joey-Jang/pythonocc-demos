import glob
import io
import os

import torch
import numpy as np
import trimesh
import cv2
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

# DinoV2 모델 로드
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").eval()


# 이미지 전처리
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# 특징 벡터 추출 및 집계
def extract_features(images, mode='mean'):
    features = []
    with torch.no_grad():
        for img in images:
            img_tensor = preprocess_image(img)
            feat = dino_model(img_tensor)
            features.append(feat.squeeze().numpy())

    features = np.array(features)
    if mode == 'mean':
        return np.mean(features, axis=0)
    elif mode == 'concat':
        return np.concatenate(features, axis=0)


if __name__ == '__main__':
    image_paths = glob.glob("./captured_views/*.png")
    vectors = extract_features(image_paths)
