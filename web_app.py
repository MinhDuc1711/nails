import base64
import os
import tempfile
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from PIL import Image

from data_visualizer import colour_code_segmentation, reverse_one_hot
from predictor import Predictor, SingleNail
from utils import get_final_path, get_validation_augmentation

app = Flask(__name__)

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '5000'))
DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'

TEMPLATE = """
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1, maximum-scale=1'>
  <title>NAILS Web App</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f7fb;
      --surface: #ffffff;
      --surface-2: #eef4fb;
      --text: #10263d;
      --muted: #5c728d;
      --accent: #1f6feb;
      --accent-2: #0f4cc3;
      --border: #dfe8f3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Segoe UI', Roboto, Arial, sans-serif;
      background: linear-gradient(135deg, var(--bg), #e8f0f8);
      color: var(--text);
      min-height: 100vh;
    }
    .app-shell { max-width: 1120px; margin: 0 auto; padding: 20px; }
    .hero {
      background: linear-gradient(120deg, var(--surface), var(--surface-2));
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(16, 38, 61, 0.08);
      margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 8px; font-size: clamp(1.7rem, 4vw, 2.3rem); letter-spacing: -0.02em; }
    .hero p { margin: 0; color: var(--muted); line-height: 1.5; }
    .board { display: grid; grid-template-columns: 1.02fr 0.98fr; gap: 18px; }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(16, 38, 61, 0.06);
    }
    .panel h2 { margin-top: 0; margin-bottom: 10px; font-size: 1.1rem; }
    form { display: grid; gap: 12px; }
    .upload-box {
      border: 2px dashed var(--border);
      border-radius: 16px;
      padding: 16px;
      background: #fbfdff;
      text-align: center;
    }
    input[type='file'] { width: 100%; padding: 10px; border-radius: 10px; background: white; border: 1px solid var(--border); color: var(--text); }
    button {
      border: 0; border-radius: 999px; padding: 12px 16px; font-weight: 700; font-size: 0.98rem;
      background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: white; cursor: pointer; min-height: 46px;
    }
    button:active { transform: scale(0.98); }
    .hint { color: var(--muted); font-size: 0.95rem; line-height: 1.4; }
    .error { color: #c0392b; font-weight: 600; margin-top: 8px; }
    .result-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 12px; }
    .result-card { background: var(--surface-2); border-radius: 16px; padding: 10px; border: 1px solid var(--border); }
    .result-card h3 { margin: 0 0 8px; font-size: 0.98rem; }
    img { width: 100%; display: block; border-radius: 12px; border: 1px solid var(--border); background: white; object-fit: contain; max-height: 320px; }
    .analysis-panel { margin-top: 18px; display: grid; gap: 14px; }
    .analysis-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: 16px; padding: 14px; }
    .rgb-chart { display: flex; align-items: flex-end; gap: 10px; height: 180px; margin-top: 10px; }
    .rgb-bar { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 8px; }
    .bar-track { width: 100%; height: 140px; display: flex; align-items: flex-end; background: #f9fbfe; border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
    .bar-fill { width: 100%; border-radius: 999px; }
    .bar-fill.red { background: linear-gradient(180deg, #ff7b7b, #d63031); }
    .bar-fill.green { background: linear-gradient(180deg, #6fdc8c, #2e8b57); }
    .bar-fill.blue { background: linear-gradient(180deg, #79b7ff, #2563eb); }
    .bar-label { font-size: 0.85rem; color: var(--muted); }
    .nail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 12px; }
    .nail-card { background: white; border: 1px solid var(--border); border-radius: 14px; padding: 10px; display: grid; gap: 8px; align-content: start; }
    .nail-card img { max-height: 120px; min-height: 90px; object-fit: cover; }
    .pill { display: inline-block; padding: 4px 8px; border-radius: 999px; background: #eaf3ff; color: var(--accent-2); font-size: 0.82rem; font-weight: 600; }
    .empty-state { color: var(--muted); padding: 22px 10px; text-align: center; border: 1px dashed var(--border); border-radius: 14px; background: #fcfeff; }
    @media (max-width: 820px) { .board { grid-template-columns: 1fr; } .result-grid, .nail-grid { grid-template-columns: 1fr; } }
    @media (max-width: 560px) { .app-shell { padding: 12px; } .hero, .panel { padding: 14px; border-radius: 16px; } button { width: 100%; } }
  </style>
</head>
<body>
  <div class='app-shell'>
    <div class='hero'>
      <h1>NAILS</h1>
      <p>Upload a nail image and the app will highlight the nail region using the original segmentation model.</p>
    </div>

    <div class='board'>
      <div class='panel'>
        <h2>Upload image</h2>
        <form method='post' enctype='multipart/form-data'>
          <div class='upload-box'>
            <input type='file' name='image' accept='image/*' required>
          </div>
          <button type='submit'>Analyze image</button>
          <div class='hint'>Tip: photos taken in good light work best. You can use your camera directly from your phone.</div>
        </form>
        {% if error %}
          <div class='error'>{{ error }}</div>
        {% endif %}
      </div>

      <div class='panel'>
        <h2>Results</h2>
        {% if result_image %}
          <div class='result-grid'>
            <div class='result-card'>
              <h3>Original image</h3>
              <img src='data:image/png;base64,{{ result_image }}' alt='Original image'>
            </div>
            <div class='result-card'>
              <h3>Segmentation overlay</h3>
              <img src='data:image/png;base64,{{ overlay_image }}' alt='Segmentation overlay'>
            </div>
          </div>
        {% else %}
          <div class='empty-state'>No image has been analyzed yet. Upload a photo to get started.</div>
        {% endif %}

        <div class='analysis-panel'>
          <div class='analysis-card'>
            <h3>Color analysis</h3>
            {% if analysis and analysis.rgb_graph %}
              <p style='margin: 4px 0 0; color: var(--muted);'>Overall RGB profile from the detected nail regions.</p>
              <div class='rgb-chart'>
                <div class='rgb-bar'>
                  <div class='bar-track'><div class='bar-fill red' style='height: {{ analysis.rgb_graph.red }}%;'></div></div>
                  <span class='bar-label'>Red</span>
                </div>
                <div class='rgb-bar'>
                  <div class='bar-track'><div class='bar-fill green' style='height: {{ analysis.rgb_graph.green }}%;'></div></div>
                  <span class='bar-label'>Green</span>
                </div>
                <div class='rgb-bar'>
                  <div class='bar-track'><div class='bar-fill blue' style='height: {{ analysis.rgb_graph.blue }}%;'></div></div>
                  <span class='bar-label'>Blue</span>
                </div>
              </div>
            {% else %}
              <div class='empty-state'>The color summary will appear here after analysis.</div>
            {% endif %}
          </div>

          <div class='analysis-card'>
            <h3>Separated nails</h3>
            {% if analysis and analysis.nails %}
              <div class='nail-grid'>
                {% for nail in analysis.nails %}
                  <div class='nail-card'>
                    <img src='data:image/png;base64,{{ nail.image }}' alt='Nail detail'>
                    <div class='pill'>{{ nail.color }}</div>
                    <div style='font-size: 0.9rem; color: var(--muted);'>RGB {{ nail.rgb }}</div>
                    <div style='font-size: 0.92rem;'>{{ nail.conclusion }}</div>
                  </div>
                {% endfor %}
              </div>
            {% else %}
              <div class='empty-state'>Only the detected nail regions will appear here.</div>
            {% endif %}
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files.get('image')
        if not image_file or not image_file.filename:
            return render_template_string(TEMPLATE, error='Please choose an image.', result_image=None, overlay_image=None)

        try:
            image_bytes = image_file.read()
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            temp_path = os.path.join(tempfile.gettempdir(), f"nails_upload_{os.getpid()}.png")
            image.save(temp_path)

            class_dict_path = get_final_path(0, ['labels', 'label_class_dict.csv'])
            class_dict = pd.read_csv(class_dict_path)
            class_names = class_dict['name'].tolist()
            class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
            select_classes = ['background', 'nail']
            select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
            select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

            predictor = Predictor(temp_path, select_class_rgb_values, select_classes, device='cpu')
            processed_img = predictor.img_preprocess()
            pred_mask = predictor.get_predicted_mask(processed_img)

            image_vis = SingleNail(temp_path, augmentation=get_validation_augmentation(), class_rgb_values=select_class_rgb_values)[0]
            image_vis = Predictor.crop_image(image_vis.astype('uint8'))
            pred_mask_colored = Predictor.crop_image(
                colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
            )
            non_black_pixels_mask = np.any(pred_mask_colored != [0, 0, 0], axis=-1)
            base_image = Image.fromarray(image_vis, 'RGB').convert('RGBA')
            mask = (np.zeros((base_image.size[1], base_image.size[0]))).astype(np.uint8)
            mask[non_black_pixels_mask] = 126
            mask = Image.fromarray(mask, mode='L')
            overlay = Image.new('RGBA', base_image.size, (255, 255, 255, 0))
            from PIL import ImageDraw
            drawing = ImageDraw.Draw(overlay)
            drawing.bitmap((0, 0), mask, fill=(10, 200, 0, 200))
            composite = Image.alpha_composite(base_image, overlay)

            analysis_image = np.array(image_vis)
            analysis = analyze_nails(analysis_image, pred_mask, select_classes.index('nail'))

            original_b64 = image_to_base64(image)
            overlay_b64 = image_to_base64(composite)
            return render_template_string(
                TEMPLATE,
                error=None,
                result_image=original_b64,
                overlay_image=overlay_b64,
                analysis=analysis,
            )
        except Exception as exc:
            return render_template_string(TEMPLATE, error=f'Analysis failed: {exc}', result_image=None, overlay_image=None)

    return render_template_string(TEMPLATE, error=None, result_image=None, overlay_image=None, analysis=None)


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


COLORS = {
    'black': (0.0, 0.0, 0.0),
    'white': (1.0, 1.0, 1.0),
    'red': (1.0, 0.0, 0.0),
    'green': (0.0, 1.0, 0.0),
    'blue': (0.0, 0.0, 1.0),
    'yellow': (1.0, 1.0, 0.0),
    'purple': (0.5, 0.0, 0.5),
    'pink': (1.0, 0.75, 0.8),
    'orange': (1.0, 0.65, 0.0),
    'brown': (0.65, 0.16, 0.16),
    'gray': (0.5, 0.5, 0.5),
}

CONCLUSIONS = {
    'black': 'Injured',
    'white': 'Liver problem',
    'red': 'Possible inflammation',
    'green': 'Healthy',
    'blue': 'Lack of oxygen',
    'yellow': 'Fungal infection',
    'purple': 'Possible circulation issue',
    'pink': 'Healthy',
    'orange': 'Possible staining',
    'brown': 'Needs attention',
    'gray': 'Possible discoloration',
}


def normalize_rgb(rgb):
    if rgb is None or len(rgb) != 3:
        return (0.0, 0.0, 0.0)
    r, g, b = rgb
    r = float(r) / 255.0 if float(r) > 1.0 else float(r)
    g = float(g) / 255.0 if float(g) > 1.0 else float(g)
    b = float(b) / 255.0 if float(b) > 1.0 else float(b)
    return (r, g, b)


def calculate_color(rgb):
    if rgb is None or len(rgb) != 3:
        return ''
    normalized_rgb = normalize_rgb(rgb)
    r, g, b = normalized_rgb
    best_name = None
    best_distance = float('inf')
    for name, (cr, cg, cb) in COLORS.items():
        distance = np.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def calculate_conclusion(rgb):
    color = calculate_color(rgb)
    if not color:
        return ''
    return CONCLUSIONS.get(color, 'Unknown')


def analyze_nails(image_rgb, pred_mask, nail_index):
    nail_channel = np.asarray(pred_mask[:, :, nail_index], dtype=np.float32)
    mask = (nail_channel > 0.5).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    if np.count_nonzero(mask) == 0:
        return {'rgb_graph': {'red': 0, 'green': 0, 'blue': 0}, 'nails': []}

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    nail_items = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 80:
            continue

        region_mask = (labels[y:y + h, x:x + w] == label)
        if region_mask.sum() < 50:
            continue

        region_pixels = image_rgb[y:y + h, x:x + w]
        region_pixels_flat = region_pixels.reshape(-1, 3)
        region_mask_flat = region_mask.reshape(-1)
        region_pixels_flat = region_pixels_flat[region_mask_flat]
        if region_pixels_flat.size == 0:
            continue

        crop_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        crop_canvas[region_mask] = region_pixels_flat.reshape(-1, 3)
        crop = Image.fromarray(crop_canvas.astype('uint8')).resize((120, 120))

        avg_color = np.round(region_pixels_flat.mean(axis=0), 1)
        normalized_avg_color = normalize_rgb(avg_color)
        nail_items.append({
            'image': image_to_base64(crop),
            'rgb': f"{normalized_avg_color[0]:.2f}, {normalized_avg_color[1]:.2f}, {normalized_avg_color[2]:.2f}",
            'color': calculate_color(avg_color).capitalize(),
            'conclusion': calculate_conclusion(avg_color),
            'x': x,
            'area': int(region_mask.sum()),
        })

    nail_items.sort(key=lambda item: item['area'], reverse=True)
    nail_items = nail_items[:5]
    nail_items.sort(key=lambda item: item['x'])
    for item in nail_items:
        item.pop('x', None)
        item.pop('area', None)

    overall_pixels = image_rgb[np.where(mask > 0)]
    if overall_pixels.size == 0:
        overall_pixels = image_rgb.reshape(-1, 3)
    overall_avg = np.round(overall_pixels.mean(axis=0), 1)
    normalized_overall_avg = normalize_rgb(overall_avg)
    rgb_graph = {
        'red': int(max(6, normalized_overall_avg[0] * 100)),
        'green': int(max(6, normalized_overall_avg[1] * 100)),
        'blue': int(max(6, normalized_overall_avg[2] * 100)),
    }

    return {'rgb_graph': rgb_graph, 'nails': nail_items}


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
