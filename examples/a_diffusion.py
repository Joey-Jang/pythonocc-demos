import torch
import torch.nn as nn
import numpy as np


class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Diffusion Process 설정
        Args:
            num_timesteps (int): 노이즈를 추가하는 단계 개수
            beta_start (float): 초기 베타 값
            beta_end (float): 마지막 베타 값
        """
        self.num_timesteps = num_timesteps

        # Beta Schedule 정의 (선형 스케줄링)
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # 누적 곱

    def add_noise(self, x0, t):
        """
        Forward Process: 입력 포인트 클라우드 x0에 노이즈 추가
        Args:
            x0 (torch.Tensor): 원본 포인트 클라우드 (B, N, 3)
            t (torch.Tensor): 타임스텝 (B,)
        Returns:
            xt (torch.Tensor): 노이즈가 추가된 포인트 클라우드 (B, N, 3)
            noise (torch.Tensor): 추가된 노이즈 (B, N, 3)
        """
        batch_size, num_points, _ = x0.shape
        noise = torch.randn_like(x0)  # 표준 정규 분포에서 노이즈 샘플링

        # 시간에 따른 alpha_bar 값 선택
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1)  # 배치 차원 맞추기
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise  # 노이즈 추가

        return xt, noise

    def sample_timesteps(self, batch_size):
        """
        랜덤한 타임스텝을 샘플링
        Args:
            batch_size (int): 배치 크기
        Returns:
            torch.Tensor: 타임스텝 인덱스 (B,)
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), dtype=torch.long)


if __name__ == "__main__":
    # 테스트
    diffusion = Diffusion(num_timesteps=1000)

    # 가짜 포인트 클라우드 데이터 생성 (Batch=2, Points=1024, XYZ=3)
    x0 = torch.randn((2, 1024, 3))

    # 랜덤 타임스텝 샘플링
    t = diffusion.sample_timesteps(batch_size=2)

    # 노이즈 추가
    xt, noise = diffusion.add_noise(x0, t)

    print(f"Original Shape: {x0.shape}")
    print(f"Noised Shape: {xt.shape}")
    print(f"Noise Shape: {noise.shape}")
