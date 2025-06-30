import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2

__all__ = [
    "PerspectiveTransform",
    "ElasticTransform",
    "ShiftScaleRotate",
    "ColorJitterCustom",
    "RandomBrightnessContrast",
    "MotionBlur",
    "RandomShadow",
    "CLAHE_approx",
    "Sharpen",
    "CannyEdge",
    "DownScale",
    "GaussianNoise",
    "GaussianBlur"
]



class PerspectiveTransform(nn.Module):
    """Applies a random perspective transformation to the image."""
    def __init__(self, scale=(0.02, 0.06), p=0.4):
        super().__init__()
        self.scale = scale
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        C, H, W = img.shape
        img_np = img.permute(1, 2, 0).numpy()
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dx = W * np.random.uniform(*self.scale, size=4)
        dy = H * np.random.uniform(*self.scale, size=4)
        dst = np.float32([[0 + dx[0], 0 + dy[0]],
                          [W - dx[1], 0 + dy[1]],
                          [W - dx[2], H - dy[2]],
                          [0 + dx[3], H - dy[3]]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img_np, matrix, (W, H))
        return torch.from_numpy(warped).permute(2, 0, 1).float()


class ElasticTransform(nn.Module):
    """Applies an elastic transformation using random displacement fields."""
    def __init__(self, alpha=0.5, sigma=15, p=0.3):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        C, H, W = img.shape
        img_np = img.permute(1, 2, 0).numpy()
        dx = cv2.GaussianBlur((np.random.rand(H, W) * 2 - 1), (0, 0), self.sigma) * self.alpha
        dy = cv2.GaussianBlur((np.random.rand(H, W) * 2 - 1), (0, 0), self.sigma) * self.alpha
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        distorted = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(distorted).permute(2, 0, 1).float()


class ShiftScaleRotate(nn.Module):
    """Random shift, scale, and rotation."""
    def __init__(self, shift_limit=0.03, scale_limit=0.05, rotate_limit=5, p=0.3):
        super().__init__()
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        angle = float(torch.empty(1).uniform_(-self.rotate_limit, self.rotate_limit))
        scale = float(torch.empty(1).uniform_(1 - self.scale_limit, 1 + self.scale_limit))
        shift_x = float(torch.empty(1).uniform_(-self.shift_limit, self.shift_limit)) * img.shape[2]
        shift_y = float(torch.empty(1).uniform_(-self.shift_limit, self.shift_limit)) * img.shape[1]
        return TF.affine(img, angle=angle, translate=(shift_x, shift_y), scale=scale, shear=0)


class ColorJitterCustom(nn.Module):
    """Color jitter for brightness, contrast, saturation, hue."""
    def __init__(self, brightness=0.03, contrast=0.05, saturation=0.07, hue=0.07, p=0.1):
        super().__init__()
        self.jitter = torch.nn.Sequential(
            TF.adjust_brightness,
            TF.adjust_contrast,
            TF.adjust_saturation,
            TF.adjust_hue
        )
        self.params = {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue
        }
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        img = TF.adjust_brightness(img, 1 + float(torch.empty(1).uniform_(-self.params['brightness'], self.params['brightness'])))
        img = TF.adjust_contrast(img, 1 + float(torch.empty(1).uniform_(-self.params['contrast'], self.params['contrast'])))
        img = TF.adjust_saturation(img, 1 + float(torch.empty(1).uniform_(-self.params['saturation'], self.params['saturation'])))
        img = TF.adjust_hue(img, float(torch.empty(1).uniform_(-self.params['hue'], self.params['hue'])))
        return img


class RandomBrightnessContrast(nn.Module):
    """Brightness and contrast adjustment."""
    def __init__(self, brightness_limit=0.03, contrast_limit=0.03, p=0.1):
        super().__init__()
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def forward(self, img):
        if torch.rand(1) >= self.p:
            return img
        brightness = 1.0 + float(torch.empty(1).uniform_(-self.brightness_limit, self.brightness_limit))
        contrast = 1.0 + float(torch.empty(1).uniform_(-self.contrast_limit, self.contrast_limit))
        mean = img.mean(dim=[1, 2], keepdim=True)
        return torch.clamp((img - mean) * contrast + mean, 0, 1) * brightness


class MotionBlur(nn.Module):
    """Simulates a simple horizontal motion blur."""
    def __init__(self, kernel_size=3, p=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.p = p

    def forward(self, img):
        if torch.rand(1) >= self.p:
            return img
        kernel = torch.ones(1, 1, 1, self.kernel_size) / self.kernel_size
        img = img.unsqueeze(0)
        return F.conv2d(img, kernel.expand(img.shape[1], -1, -1, -1),
                        padding=(0, self.kernel_size // 2), groups=img.shape[1]).squeeze(0)


class RandomShadow(nn.Module):
    """Adds a shadow-like rectangular patch."""
    def __init__(self, shadow_intensity_range=(0.01, 0.1), p=0.1):
        super().__init__()
        self.range = shadow_intensity_range
        self.p = p

    def forward(self, img):
        if torch.rand(1) >= self.p:
            return img
        C, H, W = img.shape
        x1, x2 = sorted(torch.randint(0, W, (2,)))
        y1, y2 = sorted(torch.randint(0, H, (2,)))
        shadow = 1.0 - float(torch.empty(1).uniform_(*self.range))
        img[:, y1:y2, x1:x2] *= shadow
        return img


class CLAHE_approx(nn.Module):
    """Approximates CLAHE using OpenCV."""
    def __init__(self, clip_limit=1.2, p=0.4):
        super().__init__()
        self.clip_limit = clip_limit
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit)
        for i in range(3):
            img_np[..., i] = clahe.apply(img_np[..., i])
        return torch.tensor(img_np / 255.0).permute(2, 0, 1).float()


class Sharpen(nn.Module):
    """Applies sharpening filter using convolution."""
    def __init__(self, alpha=(0.7, 0.9), lightness=(0.7, 0.9), p=0.1):
        super().__init__()
        self.alpha = alpha
        self.lightness = lightness
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        alpha = float(torch.empty(1).uniform_(*self.alpha))
        light = float(torch.empty(1).uniform_(*self.lightness))
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 9 * light, -1],
                               [-1, -1, -1]]).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(img.shape[0], 1, 1, 1)
        sharpened = F.conv2d(img.unsqueeze(0), kernel, padding=1, groups=img.shape[0])
        return (1 - alpha) * img + alpha * sharpened.squeeze(0)


class CannyEdge(nn.Module):
    """Applies OpenCV's Canny edge detector."""
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        img_np = img.permute(1, 2, 0).numpy()
        gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_tensor = torch.tensor(edges / 255.0).float().unsqueeze(0).repeat(3, 1, 1)
        return edge_tensor


class Downscale(nn.Module):
    """Simulates low-resolution downscaling and upscaling."""
    def __init__(self, scale_range=(0.9, 1.0), p=0.2):
        super().__init__()
        self.scale_range = scale_range
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        scale = float(torch.empty(1).uniform_(*self.scale_range))
        H, W = img.shape[1:]
        new_H, new_W = int(H * scale), int(W * scale)
        img = TF.resize(img, [new_H, new_W], antialias=True)
        return TF.resize(img, [H, W], antialias=True)


class GaussNoise(nn.Module):
    """Adds Gaussian noise to the image."""
    def __init__(self, std_range=(0.1, 0.2), mean=0.0, p=0.1):
        super().__init__()
        self.std_range = std_range
        self.mean = mean
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        std = float(torch.empty(1).uniform_(*self.std_range))
        noise = torch.randn_like(img) * std + self.mean
        return torch.clamp(img + noise, 0.0, 1.0)


class GaussianBlur(nn.Module):
    """Applies Gaussian blur."""
    def __init__(self, sigma_limit=(0.1, 0.2), p=0.15):
        super().__init__()
        self.sigma_limit = sigma_limit
        self.p = p

    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        sigma = float(torch.empty(1).uniform_(*self.sigma_limit))
        return TF.gaussian_blur(img, kernel_size=3, sigma=sigma)
