import base64
import os
import tempfile
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from PIL import Image

from utils import get_final_path, get_validation_augmentation

app = Flask(__name__)

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '5000'))
DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif'}

# Load the model once, at process startup, instead of on the first user
# request. Downloading + loading the model can take a long time on a slow
# connection or a small hosting instance, and doing that inside a request
# risks the request timing out (server kills the connection mid-response,
# which shows up in the browser as "Unexpected end of JSON input"). Loading
# it here means that work happens during deploy/boot, where hosts like
# Render allow a much longer grace period before a health check gives up.
_predictor_assets = None
_predictor_load_error = None


def get_predictor_assets():
    """Return the cached (model, preprocessing_fn) tuple, loading it lazily
    only as a fallback if startup loading failed or hasn't finished yet."""
    global _predictor_assets, _predictor_load_error
    if _predictor_assets is not None:
        return _predictor_assets
    if _predictor_load_error is not None:
        raise _predictor_load_error
    from predictor import Predictor
    _predictor_assets = Predictor.load_assets(device='cpu')
    return _predictor_assets


def _load_predictor_assets_at_startup():
    global _predictor_assets, _predictor_load_error
    try:
        from predictor import Predictor
        print('Loading model assets at startup...')
        _predictor_assets = Predictor.load_assets(device='cpu')
        print('Model assets loaded successfully.')
    except Exception as exc:  # noqa: BLE001
        # Don't crash the whole process if this fails — keep the server up
        # so it can still respond with a clear JSON error, and retry lazily
        # on the next request.
        import traceback
        traceback.print_exc()
        _predictor_load_error = exc


_load_predictor_assets_at_startup()


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    image_file = request.files.get('image')
    if not image_file or not image_file.filename:
        return jsonify(error='Please choose an image.'), 400

    if not allowed_file(image_file.filename):
        return jsonify(error='Only PNG, JPG, JPEG, and JFIF image types are accepted.'), 400

    temp_path = None
    try:
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        image.save(temp_path)

        from data_visualizer import colour_code_segmentation, reverse_one_hot
        from predictor import Predictor, SingleNail

        class_dict_path = get_final_path(0, ['labels', 'label_class_dict.csv'])
        class_dict = pd.read_csv(class_dict_path)
        class_names = class_dict['name'].tolist()
        class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
        select_classes = ['background', 'nail']
        select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
        select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

        from predictor import Predictor
        model, preprocessing_fn = get_predictor_assets()

        predictor = Predictor(
            temp_path,
            select_class_rgb_values,
            select_classes,
            device='cpu',
            model=model,
            preprocessing_fn=preprocessing_fn,
        )
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

        return jsonify(
            result_image=image_to_base64(image),
            overlay_image=image_to_base64(composite),
            analysis=analysis,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        message = str(exc)
        if not message:
            message = 'The analysis pipeline failed while processing the image.'
        return jsonify(error=message), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


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
