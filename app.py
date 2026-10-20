"""
app.py
------
Сучасний вебінтерфейс (Gradio) для монокулярної карти глибини
на основі Depth Anything V2.

Запуск:
  python app.py
  (відкриється за адресою http://127.0.0.1:7860)
"""

import os
import sys
from pathlib import Path

# Забезпечуємо коректне кодування для Windows консолі
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import shutil
import tempfile
import zipfile
from typing import List, Tuple, Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from depth_engine import (
    DepthAnythingV2Engine,
    MODELS,
    COLORMAPS,
    DEFAULT_MODEL_KEY,
)

# Створюємо директорію для сесійних файлів
CACHE_DIR = Path("./web_outputs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ініціалізація рушія
engine = DepthAnythingV2Engine()
device_info = engine.get_device_info()


def process_single_image(
    image: Optional[Image.Image],
    model_choice: str,
    colormap: str,
    invert_depth: bool,
    enable_bokeh: bool,
    focus_depth: float,
    blur_strength: int,
    enable_wiggle: bool,
    wiggle_shift: int,
    enable_3d: bool,
    mesh_resolution: int,
    progress=gr.Progress(),
):
    """Обробляє одне завантажене фото"""
    if image is None:
        raise gr.Error("Будь ласка, завантажте фотографію для обробки!")

    progress(0.1, desc="Аналіз фотографії та завантаження моделі...")
    
    # 1. Передбачення карти глибини
    res = engine.estimate_depth(
        image,
        model_name=model_choice,
        invert=invert_depth,
    )
    depth_norm = res["depth_norm"]
    orig_rgb = res["original_rgb"]
    depth_u16 = res["depth_uint16"]

    progress(0.4, desc="Колірне картування та збереження карт...")
    color_depth_rgb = engine.colorize_depth(depth_norm, colormap=colormap)
    color_depth_pil = Image.fromarray(color_depth_rgb)

    # Збереження файлів для завантаження
    session_id = next(tempfile._get_candidate_names())
    session_folder = CACHE_DIR / session_id
    session_folder.mkdir(parents=True, exist_ok=True)

    color_png_path = str(session_folder / f"depth_{colormap}.png")
    color_depth_pil.save(color_png_path)

    raw16_png_path = str(session_folder / "depth_raw_16bit.png")
    cv2.imwrite(raw16_png_path, depth_u16)

    # 2. Боке / Портретний режим
    bokeh_output = None
    if enable_bokeh:
        progress(0.6, desc="Симуляція оптичного боке...")
        bokeh_rgb = engine.apply_bokeh(
            orig_rgb,
            depth_norm,
            focus_depth=focus_depth,
            depth_of_field=0.15,
            blur_strength=blur_strength,
        )
        bokeh_output = Image.fromarray(bokeh_rgb)

    # 3. 3D Wiggle GIF
    wiggle_output_path = None
    if enable_wiggle:
        progress(0.75, desc="Генерація 3D Wiggle паралакс-анімації...")
        gif_file = str(session_folder / "wiggle_3d.gif")
        engine.create_wiggle_gif(
            orig_rgb,
            depth_norm,
            gif_file,
            max_shift_px=wiggle_shift,
            num_frames=14,
            fps=14,
        )
        wiggle_output_path = gif_file

    # 4. 3D Mesh / Pointcloud
    mesh_path = None
    ply_path = None
    if enable_3d:
        progress(0.85, desc="Побудова 3D моделі (.obj та .ply)...")
        mesh_file = str(session_folder / "depth_mesh.obj")
        engine.export_3d_mesh_obj(
            orig_rgb,
            depth_norm,
            mesh_file,
            grid_size=mesh_resolution,
            depth_scale=0.35,
        )
        mesh_path = mesh_file

        ply_file = str(session_folder / "depth_pointcloud.ply")
        engine.export_pointcloud_ply(
            orig_rgb,
            depth_norm,
            ply_file,
            max_points=80000,
        )
        ply_path = ply_file

    progress(1.0, desc="Готово!")

    # Повертаємо:
    # 1. slider comparison: (original, colored depth)
    # 2. colored depth image
    # 3. 16-bit depth file path
    # 4. colored depth file path
    # 5. bokeh image
    # 6. wiggle gif path
    # 7. 3d model path (obj)
    # 8. 3d ply path
    return (
        (image, color_depth_pil),
        color_depth_pil,
        color_png_path,
        raw16_png_path,
        bokeh_output,
        wiggle_output_path,
        mesh_path,
        ply_path,
    )


def process_batch_images(
    files: List[str],
    model_choice: str,
    colormap: str,
    invert_depth: bool,
    save_raw16: bool,
    save_pointcloud: bool,
    progress=gr.Progress(),
):
    """Пакетна обробка списку зображень з генерацією ZIP-архіву"""
    if not files:
        raise gr.Error("Будь ласка, завантажте хоча б одне фото для пакетної обробки!")

    batch_id = next(tempfile._get_candidate_names())
    batch_dir = CACHE_DIR / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    gallery_items = []
    total = len(files)

    for i, file_obj in enumerate(files):
        progress((i / total), desc=f"Обробка {i+1}/{total}...")
        file_path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
        try:
            pil_img = Image.open(file_path)
            res = engine.estimate_depth(pil_img, model_name=model_choice, invert=invert_depth)
            d_norm = res["depth_norm"]
            c_rgb = engine.colorize_depth(d_norm, colormap=colormap)

            stem = file_path.stem
            # Зберігаємо кольорову карту
            color_path = batch_dir / f"{stem}_depth_{colormap}.png"
            Image.fromarray(c_rgb).save(color_path)
            gallery_items.append(str(color_path))

            # 16-біт
            if save_raw16:
                raw_path = batch_dir / f"{stem}_depth_raw16.png"
                cv2.imwrite(str(raw_path), res["depth_uint16"])

            # Хмара точок
            if save_pointcloud:
                ply_path = batch_dir / f"{stem}_pointcloud.ply"
                engine.export_pointcloud_ply(res["original_rgb"], d_norm, str(ply_path), max_points=60000)

        except Exception as e:
            print(f"Помилка при обробці {file_path}: {e}")

    # Створюємо ZIP архів
    progress(0.95, desc="Створення ZIP архіву результатів...")
    zip_path = CACHE_DIR / f"batch_depth_results_{batch_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, filenames in os.walk(batch_dir):
            for fn in filenames:
                fp = Path(root) / fn
                zipf.write(fp, arcname=fn)

    progress(1.0, desc="Пакетну обробку завершено!")
    return gallery_items, str(zip_path)


# ==========================================
# Інтерфейс користувача Gradio
# ==========================================
custom_css = """
.main-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    padding: 24px;
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.4);
}
.badge-gpu {
    display: inline-block;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #6ee7b7;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 8px;
}
.btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39) !important;
    transition: all 0.2s ease-in-out !important;
}
.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5) !important;
}
.card-box {
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
"""

with gr.Blocks(title="Depth Anything V2 — Monocular Depth Studio") as demo:
    # Заголовок
    gpu_badge = (
        f"⚡ GPU: {device_info['gpu_name']} | VRAM: {device_info.get('vram_total_mb', 'N/A')} MB | "
        f"PyTorch: {device_info['torch_version']} (CUDA: {device_info['cuda_available']})"
    )
    gr.HTML(
        f"""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;">
                🗺️ Depth Anything V2 — Студія Монокулярної Карти Глибини
            </h1>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
                Штучний інтелект нового покоління для видобування 3D простору з будь-якого 2D фото.
            </p>
            <div class="badge-gpu">{gpu_badge}</div>
        </div>
        """
    )

    with gr.Tabs():
        # ------------------------------------------------------------------
        # Вкладка 1: Одиночне фото
        # ------------------------------------------------------------------
        with gr.TabItem("📸 Одиночне фото (Studio)"):
            with gr.Row():
                # Ліва колонка: Вхідні дані та керування
                with gr.Column(scale=5):
                    input_img = gr.Image(
                        label="Вхідне фото",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )

                    with gr.Accordion("⚙️ Налаштування нейромережі та відображення", open=True):
                        model_dropdown = gr.Dropdown(
                            label="Модель Depth Anything V2",
                            choices=list(MODELS.keys()),
                            value=DEFAULT_MODEL_KEY,
                            info="Small (найшвидша), Base (збалансована), Large (максимальна деталізація)",
                        )
                        colormap_dropdown = gr.Dropdown(
                            label="Колірна палітра глибини",
                            choices=COLORMAPS,
                            value="inferno",
                            info="Схема кольорів для візуалізації (Inferno, Turbo, Viridis тощо)",
                        )
                        invert_checkbox = gr.Checkbox(
                            label="Інвертувати глибину",
                            value=False,
                            info="За замовчуванням світле/тепліше = ближче до камери",
                        )

                    with gr.Accordion("✨ Додаткові 3D та фото-ефекти", open=True):
                        bokeh_check = gr.Checkbox(
                            label="📷 Симуляція оптичного боке (Portrait Bokeh)",
                            value=True,
                            info="Розмиття фону на основі глибини різкості (DoF)",
                        )
                        with gr.Row(visible=True):
                            focus_slider = gr.Slider(
                                label="Площина фокусування",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.6,
                                step=0.02,
                                info="0.0 = фокус на фоні, 1.0 = фокус на передньому плані",
                            )
                            blur_slider = gr.Slider(
                                label="Сила розмиття",
                                minimum=3,
                                maximum=35,
                                value=15,
                                step=2,
                            )

                        wiggle_check = gr.Checkbox(
                            label="🎞️ 3D Wiggle паралакс-анімація (GIF)",
                            value=True,
                            info="Створює ефект живого об'ємного 3D похитування",
                        )
                        wiggle_shift_slider = gr.Slider(
                            label="Амплітуда 3D коливання (px)",
                            minimum=4,
                            maximum=26,
                            value=12,
                            step=1,
                        )

                        mesh_check = gr.Checkbox(
                            label="🧊 Побудова 3D Моделі (Mesh / Pointcloud)",
                            value=True,
                            info="Генерує .obj полігональну сітку для інтерактивного 3D перегляду та .ply хмару",
                        )
                        mesh_res_slider = gr.Slider(
                            label="Роздільна здатність 3D мешу",
                            minimum=80,
                            maximum=240,
                            value=150,
                            step=10,
                            info="Кількість полігонів по горизонталі",
                        )

                    process_btn = gr.Button(
                        "🚀 Обробити фото (Generate Depth)",
                        variant="primary",
                        elem_classes=["btn-primary"],
                        size="lg",
                    )

                # Права колонка: Результати
                with gr.Column(scale=7):
                    gr.Markdown("### 🔍 Інтерактивне порівняння")
                    compare_slider = gr.ImageSlider(
                        label="Оригінал (ліворуч) ↔ Карта глибини (праворуч)",
                        type="pil",
                    )

                    with gr.Tabs():
                        with gr.TabItem("🎨 Карта глибини"):
                            depth_preview = gr.Image(label="Карта глибини (PNG)", type="pil")
                            with gr.Row():
                                download_color = gr.File(label="Завантажити кольорову карту (PNG)")
                                download_raw16 = gr.File(label="Завантажити 16-бітну карту (для 3D/VFX)")

                        with gr.TabItem("📷 Портретне боке"):
                            bokeh_preview = gr.Image(label="Фото з глибиною різкості (DoF)", type="pil")

                        with gr.TabItem("🎞️ 3D Wiggle GIF"):
                            wiggle_preview = gr.Image(label="3D стереоскопічна анімація", type="filepath")

                        with gr.TabItem("🧊 3D Переглядач"):
                            gr.Markdown("Обертайте та наближайте згенеровану 3D модель мишкою:")
                            model_3d_preview = gr.Model3D(label="3D Mesh (.obj)", height=420)
                            with gr.Row():
                                download_ply = gr.File(label="Завантажити хмару точок (.ply)")

            # Зв'язування подій
            process_btn.click(
                fn=process_single_image,
                inputs=[
                    input_img,
                    model_dropdown,
                    colormap_dropdown,
                    invert_checkbox,
                    bokeh_check,
                    focus_slider,
                    blur_slider,
                    wiggle_check,
                    wiggle_shift_slider,
                    mesh_check,
                    mesh_res_slider,
                ],
                outputs=[
                    compare_slider,
                    depth_preview,
                    download_color,
                    download_raw16,
                    bokeh_preview,
                    wiggle_preview,
                    model_3d_preview,
                    download_ply,
                ],
            )

        # ------------------------------------------------------------------
        # Вкладка 2: Пакетна обробка
        # ------------------------------------------------------------------
        with gr.TabItem("📁 Пакетна обробка (Batch Processing)"):
            gr.Markdown(
                """
                ### 📦 Пакетна генерація карт глибини
                Завантажте серію фотографій (або цілу папку) для одночасної обробки. Всі карти глибини
                будуть згенеровані та спаковані у єдиний **ZIP-архів**.
                """
            )
            with gr.Row():
                with gr.Column(scale=4):
                    batch_files = gr.File(
                        label="Виберіть фотографії для пакетної обробки",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    batch_model = gr.Dropdown(
                        label="Модель",
                        choices=list(MODELS.keys()),
                        value=DEFAULT_MODEL_KEY,
                    )
                    batch_cmap = gr.Dropdown(
                        label="Колірна палітра",
                        choices=COLORMAPS,
                        value="inferno",
                    )
                    batch_invert = gr.Checkbox(label="Інвертувати глибину", value=False)
                    batch_save_16 = gr.Checkbox(label="Зберігати 16-бітні карти (.png)", value=True)
                    batch_save_ply = gr.Checkbox(label="Зберігати 3D хмари точок (.ply)", value=False)

                    batch_btn = gr.Button(
                        "⚡ Запустити пакетну обробку",
                        variant="primary",
                        elem_classes=["btn-primary"],
                    )

                with gr.Column(scale=6):
                    batch_zip_download = gr.File(label="Завантажити ZIP-архів з усіма картами")
                    batch_gallery = gr.Gallery(
                        label="Галерея згенерованих карт глибини",
                        columns=3,
                        height="auto",
                    )

            batch_btn.click(
                fn=process_batch_images,
                inputs=[
                    batch_files,
                    batch_model,
                    batch_cmap,
                    batch_invert,
                    batch_save_16,
                    batch_save_ply,
                ],
                outputs=[batch_gallery, batch_zip_download],
            )

        # ------------------------------------------------------------------
        # Вкладка 3: Довідка та Налаштування
        # ------------------------------------------------------------------
        with gr.TabItem("ℹ️ Довідка та Інструкції"):
            gr.Markdown(
                f"""
                ### 🎯 Про проект Depth Anything V2
                **Depth Anything V2** — це найсучасніша модель для монокулярної оцінки глибини (Monocular Depth Estimation), розроблена дослідниками з TikTok та University of Hong Kong.
                
                Вона здатна з **одного звичайного 2D фото** створити неперевершено точну, плавну та деталізовану просторову геометрію навіть для дрібних деталей (волосся, листя, тонкі прути, скло).

                ---

                ### 🛠️ Сфери застосування:
                1. **3D Моделювання та Скульптури (Blender / ZBrush / Cinema4D)**:
                   - Завантажуйте **16-бітну карту глибини (RAW 16-bit PNG)** як *Displacement / Height Map* модифікатор на площину (Subdivision Surface).
                   - За секунди отримуйте 3D барельєф та повноцінну 3D сцену.
                2. **VFX та Пост-продакшн (After Effects, Nuke, DaVinci Resolve)**:
                   - Маски глибини (Z-Depth passes) для додавання атмосферного туману, диму, об'ємного освітлення (God Rays) та розмиття заднього плану.
                3. **Геймдев (Unreal Engine 5 / Unity)**:
                   - Генерація 3D ассетів, parallax occlusion mapping та перетворення концепт-артів у 3D рівні.
                4. **3D Друк (Lithophane / 3D Print)**:
                   - Експорт `.ply` або `.obj` безпосередньо у слайсер (Cura / PrusaSlicer) для створення літофаній та тактильних барельєфів.

                ---

                ### 💻 Ваше апаратне забезпечення:
                - **Графічний адаптер**: `{device_info['gpu_name']}`
                - **Статус CUDA**: `{'Увімкнено (прискорення активно)' if device_info['cuda_available'] else 'Вимкнено (CPU mode)'}`
                - **Відеопам'ять (VRAM)**: `{device_info.get('vram_total_mb', 'N/A')} MB`
                - **Версія PyTorch**: `{device_info['torch_version']}`

                ---

                ### 💾 Де зберігаються моделі на комп'ютері (Кеш):
                - **Depth Anything (Hugging Face)**: `{DepthAnythingV2Engine.get_cache_paths()['huggingface_cache']}`
                - **MiDaS (PyTorch Hub)**: `{DepthAnythingV2Engine.get_cache_paths()['torch_cache']}`
                - *Завантажені моделі кешуються один раз і надалі працюють повністю офлайн без повторного завантаження.*
                """
            )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Запуск Depth Anything V2 Web UI...")
    print("  Адреса сервера: http://127.0.0.1:7860")
    print("=" * 60 + "\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=custom_css,
    )
