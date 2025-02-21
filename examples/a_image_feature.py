import glob

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModel

# DinoV2 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_model.eval().to(device)

# 이미지 변환 (DinoV2 input size)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features(image_paths, mode="mean"):
    """ 멀티뷰 이미지에서 DinoV2 특징 벡터 추출 """
    vectors = []

    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            feat = dinov2_model(image)  # 특징 벡터 추출
            print(feat.shape)
            feat = feat.mean(dim=[1, 2])  # 공간 평균 풀링
            vectors.append(feat)

    vectors = torch.stack(vectors, dim=0)

    if mode == "mean":
        return vectors.mean(dim=0)  # 평균 벡터
    elif mode == "concat":
        return vectors.view(-1)  # 벡터 연결

    return vectors  # 원본 벡터 리스트 반환


if __name__ == "__main__":
    image_paths = glob.glob("./captured_views/*.png")
    vectors = extract_features(image_paths)
