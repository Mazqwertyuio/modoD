"""
depth_engine.py
---------------
Модуль для роботи з моделлю Depth Anything V2:
- Монокулярна оцінка глибини з фотографій (Small, Base, Large)
- Генерація нормалізованих карт глибини та 16-бітних карт для 3D/VFX
- Колірне картування (Inferno, Viridis, Turbo, Plasma, Magma тощо)
- Створення 3D хмари точок (.ply) та 3D полігонального мешу (.obj)
- Симуляція глибини різкості (Bokeh / Depth-of-Field портретний ефект)
- Генерація стереоскопічної 3D Wiggle GIF анімації (паралакс-ефект)
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib
import matplotlib.cm as cm
from transformers import pipeline

# Доступні моделі Depth Anything V2, V1 та MiDaS
MODELS = {
    # Depth Anything V2 (Рекомендовано)
    "Depth Anything V2 - Small (~25M, найшвидша)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V2 - Base (~97M, збалансована)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 - Large (~335M, найвища точність)": "depth-anything/Depth-Anything-V2-Large-hf",
    # Depth Anything V1
    "Depth Anything V1 - Small (~25M)": "LiheYoung/depth-anything-small-hf",
    "Depth Anything V1 - Base (~97M)": "LiheYoung/depth-anything-base-hf",
    # MiDaS
    "MiDaS v2.1 - Small (~21M, дуже швидка)": "midas:MiDaS_small",
    "MiDaS v3.0 - DPT Hybrid (~120M)": "midas:DPT_Hybrid",
    "MiDaS v3.0 - DPT Large (~345M)": "midas:DPT_Large",
}

DEFAULT_MODEL_KEY = "Depth Anything V2 - Small (~25M, найшвидша)"

COLORMAPS = [
    "inferno",
    "magma",
    "plasma",
    "viridis",
    "turbo",
    "cividis",
    "spectral",
    "twilight",
    "gist_earth",
    "gray"
]


class DepthAnythingV2Engine:
    """
    Клас-обгортка для роботи з Depth Anything V2, V1 та MiDaS:
    забезпечує кешування моделі, вибір пристрою (CUDA/CPU) та набір пост-обробок.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.current_model_key: Optional[str] = None
        self.pipeline = None
        self.midas_model = None
        self.midas_transform = None

    @staticmethod
    def get_cache_paths() -> Dict[str, Any]:
        """Повертає шляхи, де зберігаються моделі на комп'ютері, та їх розмір"""
        hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub")
        torch_home = os.environ.get("TORCH_HOME") or os.path.expanduser("~/.cache/torch/hub")

        def get_dir_size_mb(path: str) -> float:
            total = 0
            if os.path.exists(path):
                for dirpath, _, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        try:
                            total += os.path.getsize(fp)
                        except OSError:
                            pass
            return round(total / (1024 * 1024), 1)

        hf_models = []
        if os.path.exists(hf_home):
            for item in os.listdir(hf_home):
                if item.startswith("models--"):
                    hf_models.append(item.replace("models--", "").replace("--", "/"))

        torch_checkpoints = []
        chk_dir = os.path.join(torch_home, "checkpoints")
        if os.path.exists(chk_dir):
            torch_checkpoints = os.listdir(chk_dir)

        return {
            "huggingface_cache": os.path.abspath(hf_home),
            "huggingface_size_mb": get_dir_size_mb(hf_home),
            "huggingface_downloaded": hf_models,
            "torch_cache": os.path.abspath(torch_home),
            "torch_size_mb": get_dir_size_mb(torch_home),
            "torch_checkpoints": torch_checkpoints,
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Повертає інформацію про поточний пристрій та GPU"""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Немає (CPU)",
            "torch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            vram_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            info["vram_total_mb"] = round(vram_total, 1)
            info["vram_allocated_mb"] = round(vram_allocated, 1)
        return info

    def load_model(self, model_key_or_name: str):
        """Завантажує обрану модель (HuggingFace або MiDaS через torch.hub)"""
        target = MODELS.get(model_key_or_name, model_key_or_name)
        if self.current_model_key == target:
            return

        print(f"[*] Завантаження моделі: {model_key_or_name} ({target})...")

        if target.startswith("midas:"):
            midas_type = target.split(":", 1)[1]
            # Завантаження MiDaS через torch.hub
            self.midas_model = torch.hub.load("intel-isl/MiDaS", midas_type).to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if midas_type == "MiDaS_small":
                self.midas_transform = midas_transforms.small_transform
            else:
                self.midas_transform = midas_transforms.dpt_transform

            self.pipeline = None
            self.current_model_key = target
        else:
            # Завантаження Depth Anything V2 / V1 через transformers pipeline
            device_arg = 0 if "cuda" in self.device else -1
            self.pipeline = pipeline(
                task="depth-estimation",
                model=target,
                device=device_arg,
            )
            self.midas_model = None
            self.midas_transform = None
            self.current_model_key = target

        print(f"[OK] Модель {target} успішно активована на {self.device}!")

    def estimate_depth(
        self,
        image: Image.Image,
        model_name: str = DEFAULT_MODEL_KEY,
        invert: bool = False
    ) -> Dict[str, Any]:
        """
        Виконує передбачення глибини для вхідного зображення.

        Повертає словник:
        - 'depth_norm': float32 масив у діапазоні [0.0, 1.0] (за замовчуванням 1.0 = близько, 0.0 = далеко)
        - 'depth_raw': вихідний тензор моделі
        - 'depth_uint16': 16-бітний масив [0, 65535] для професійного 3D/VFX
        - 'depth_uint8': 8-бітний масив [0, 255]
        - 'original_rgb': np.ndarray (RGB) оригінального зображення
        """
        self.load_model(model_name)

        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size

        if self.midas_model is not None:
            # Інференс для моделей MiDaS
            img_np = np.array(image)
            input_batch = self.midas_transform(img_np).to(self.device)
            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth_arr = prediction.detach().cpu().numpy()
        else:
            # Інференс для моделей Depth Anything (V1 / V2)
            res = self.pipeline(image)
            pred_depth = res["predicted_depth"]
            if isinstance(pred_depth, torch.Tensor):
                depth_arr = pred_depth.detach().cpu().numpy()
            else:
                depth_arr = np.array(pred_depth, dtype=np.float32)

            if depth_arr.shape != (orig_h, orig_w):
                depth_arr = cv2.resize(
                    depth_arr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
                )

        # Нормалізація в [0, 1]
        d_min, d_max = depth_arr.min(), depth_arr.max()
        if d_max > d_min:
            depth_norm = (depth_arr - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_arr, dtype=np.float32)

        if invert:
            depth_norm = 1.0 - depth_norm

        depth_uint16 = (depth_norm * 65535.0).clip(0, 65535).astype(np.uint16)
        depth_uint8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
        orig_rgb = np.array(image, dtype=np.uint8)

        return {
            "depth_norm": depth_norm.astype(np.float32),
            "depth_raw": depth_arr,
            "depth_uint16": depth_uint16,
            "depth_uint8": depth_uint8,
            "original_rgb": orig_rgb,
            "size": (orig_w, orig_h),
        }

    @staticmethod
    def colorize_depth(depth_norm: np.ndarray, colormap: str = "inferno") -> np.ndarray:
        """
        Застосовує колірну шкалу до нормалізованої карти глибини [0.0, 1.0].
        Повертає RGB np.ndarray (H, W, 3) uint8.
        """
        cmap_name = colormap.lower()
        if cmap_name == "gray" or cmap_name == "grayscale":
            gray_u8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
            return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        try:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = matplotlib.colormaps.get_cmap("inferno")

        colored = cmap(depth_norm)[:, :, :3]  # RGB, діапазон [0, 1]
        return (colored * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def apply_bokeh(
        image_rgb: np.ndarray,
        depth_norm: np.ndarray,
        focus_depth: float = 0.5,
        depth_of_field: float = 0.15,
        blur_strength: int = 15,
    ) -> np.ndarray:
        """
        Створює реалістичний ефект розмиття фону / портретного боке:
        - focus_depth: глибина площини різкості [0.0, 1.0]
        - depth_of_field: товщина зони різкості [0.02, 0.5]
        - blur_strength: радіус розмиття об'єктиву
        """
        if blur_strength <= 0:
            return image_rgb.copy()

        # Розрахунок відстані від площини фокусування
        diff = np.abs(depth_norm - focus_depth)

        # Коефіцієнт розмиття (0 - різко, 1 - максимальне розмиття)
        dof_safe = max(depth_of_field, 0.01)
        blur_factor = np.clip((diff - dof_safe) / (1.0 - dof_safe + 1e-6), 0.0, 1.0)
        # Згладжуємо перехід
        blur_factor = cv2.GaussianBlur(blur_factor, (7, 7), 0)
        blur_mask_3ch = np.repeat(blur_factor[:, :, np.newaxis], 3, axis=2)

        # Кількарівневе якісне розмиття
        ksize = blur_strength * 2 + 1
        blurred = cv2.GaussianBlur(image_rgb, (ksize, ksize), 0)

        # Змішування різкого та розмитого зображення
        result = (image_rgb.astype(np.float32) * (1.0 - blur_mask_3ch) +
                  blurred.astype(np.float32) * blur_mask_3ch)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def create_wiggle_gif(
        image_rgb: np.ndarray,
        depth_norm: np.ndarray,
        output_path: str,
        max_shift_px: int = 14,
        num_frames: int = 14,
        fps: int = 14,
    ) -> str:
        """
        Створює стереоскопічну 3D wiggle GIF-анімацію (ефект паралаксу).
        """
        h, w = depth_norm.shape
        frames: List[Image.Image] = []

        # Базова сітка координат
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        x_coords = x_coords.astype(np.float32)
        y_coords = y_coords.astype(np.float32)

        # Центруємо глибину навколо 0.5
        disparity = depth_norm.astype(np.float32) - 0.5

        for i in range(num_frames):
            phase = 2.0 * np.pi * (i / num_frames)
            shift = max_shift_px * np.sin(phase)

            # Розрахунок зсуву для кожного пікселя залежно від його глибини
            map_x = x_coords - shift * disparity
            map_y = y_coords

            warped = cv2.remap(
                image_rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            frames.append(Image.fromarray(warped))

        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        return output_path

    @staticmethod
    def export_pointcloud_ply(
        image_rgb: np.ndarray,
        depth_norm: np.ndarray,
        output_path: str,
        max_points: int = 120000,
        focal_scale: float = 1.0,
        depth_scale: float = 2.5,
    ) -> str:
        """
        Створює кольорову 3D хмару точок у форматі .PLY:
        - image_rgb: кольори для кожної точки
        - depth_norm: карта глибини
        - output_path: шлях до .ply файлу
        """
        h, w = depth_norm.shape
        total_pixels = h * w

        # Крок проріджування для оптимізації розміру
        step = max(1, int(np.sqrt(total_pixels / max_points)))
        sub_depth = depth_norm[::step, ::step]
        sub_rgb = image_rgb[::step, ::step]

        sub_h, sub_w = sub_depth.shape

        u, v = np.meshgrid(np.arange(sub_w), np.arange(sub_h))
        u = u.astype(np.float32) * step
        v = v.astype(np.float32) * step

        cx = w / 2.0
        cy = h / 2.0
        focal = max(w, h) * focal_scale

        # Z глибина: чим більше значення depth_norm (ближче), тим менша відстань
        # Додаємо безпечну базу, щоб уникнути ділення на 0
        z = (1.5 - sub_depth * 0.9) * depth_scale
        x = (u - cx) * z / focal
        y = (v - cy) * z / focal

        points = np.stack([x.ravel(), -y.ravel(), -z.ravel()], axis=-1)
        colors = sub_rgb.reshape(-1, 3)

        # Запис у бінарний або ascii PLY формат
        num_points = points.shape[0]
        header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            for i in range(num_points):
                f.write(
                    f"{points[i, 0]:.4f} {points[i, 1]:.4f} {points[i, 2]:.4f} "
                    f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
                )

        return output_path

    @staticmethod
    def export_3d_mesh_obj(
        image_rgb: np.ndarray,
        depth_norm: np.ndarray,
        output_path: str,
        grid_size: int = 160,
        depth_scale: float = 0.35,
    ) -> str:
        """
        Створює 3D меш (.obj) з полігональною сіткою та кольорами вершин
        для швидкого інтерактивного перегляду у браузері / Blender.
        """
        orig_h, orig_w = depth_norm.shape
        aspect = orig_h / orig_w
        mesh_w = grid_size
        mesh_h = max(20, int(grid_size * aspect))

        d_small = cv2.resize(depth_norm, (mesh_w, mesh_h), interpolation=cv2.INTER_AREA)
        rgb_small = cv2.resize(image_rgb, (mesh_w, mesh_h), interpolation=cv2.INTER_AREA)

        # Координати X [-1, 1], Y [-aspect, aspect], Z
        xs = np.linspace(-1.0, 1.0, mesh_w)
        ys = np.linspace(aspect, -aspect, mesh_h)
        xx, yy = np.meshgrid(xs, ys)

        # Ближчі об'єкти (depth_norm -> 1) мають більшу висоту Z
        zz = d_small * depth_scale

        verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        colors = rgb_small.reshape(-1, 3) / 255.0

        # Створення полігональних граней (трикутників)
        faces = []
        for r in range(mesh_h - 1):
            for c in range(mesh_w - 1):
                idx = r * mesh_w + c
                # Два трикутники на квадрат
                faces.append([idx + 1, idx + 2, idx + mesh_w + 1])
                faces.append([idx + 2, idx + mesh_w + 2, idx + mesh_w + 1])

        # Запис OBJ файлу
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Depth Anything V2 3D Mesh\n")
            # Вершини з кольором (x y z r g b)
            for v, c in zip(verts, colors):
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            # Грані (1-based index)
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        return output_path
