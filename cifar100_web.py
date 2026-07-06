"""
CIFAR-100 PyTorch web demo (Flask).

Run:
    python cifar100_web.py

Then open: http://127.0.0.1:5000
"""

import base64
import io
import os
import sys

import torch
import numpy as np
from flask import Flask, make_response, redirect, render_template_string, request, session, url_for
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "cifar100_pytorch"))

from models import get_model  # noqa: E402

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "cifar100-local-demo-key")

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "cifar-100-python")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_cache = {}


def find_latest_model(task):
    if not os.path.isdir(MODELS_DIR):
        return None
    prefix = f"{task}_"
    files = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith(prefix) and f.endswith("_best.pth")
    ]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(MODELS_DIR, files[0])


def load_task_model(task, force_reload=False):
    model_path = find_latest_model(task)
    if model_path is None:
        raise FileNotFoundError(
            f"No {task} model found in {MODELS_DIR}. "
            f"Run training first: python cifar100_pytorch/train.py --label {task}"
        )

    cache_key = f"{model_path}:{os.path.getmtime(model_path)}"
    if not force_reload and task in _model_cache:
        cached = _model_cache[task]
        if cached.get("cache_key") == cache_key:
            return cached

    ckpt = torch.load(model_path, map_location=DEVICE)
    num_classes = ckpt.get("num_classes", 100 if task == "fine" else 20)
    class_names = ckpt.get("class_names")
    if class_names is None:
        raise ValueError(f"Checkpoint missing class_names: {model_path}")

    model = get_model("resnet", num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _model_cache[task] = {
        "model": model,
        "class_names": class_names,
        "path": model_path,
        "num_classes": num_classes,
        "cache_key": cache_key,
    }
    return _model_cache[task]


def preprocess(image):
    """Match training pipeline: float32 pixels in [0, 1]."""
    image = image.convert("RGB").resize((32, 32))
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor, image


@torch.no_grad()
def predict(task, tensor, top_k=5):
    bundle = load_task_model(task)
    model = bundle["model"]
    class_names = bundle["class_names"]

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    conf, pred_idx = torch.max(probs, 0)

    top_probs, top_idx = torch.topk(probs, min(top_k, len(class_names)))
    top_list = [
        {
            "name": class_names[i.item()],
            "prob": float(p),
            "pct": f"{float(p) * 100:.2f}%",
        }
        for p, i in zip(top_probs, top_idx)
    ]

    return class_names[pred_idx.item()], float(conf), top_list, bundle["path"]


HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CIFAR-100 分类</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            background: #f5f7fb;
            color: #1f2937;
        }
        .wrap {
            max-width: 640px;
            margin: 40px auto;
            padding: 0 16px;
        }
        .card {
            background: #fff;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        }
        h1 { margin: 0 0 20px; font-size: 24px; text-align: center; }
        label { font-weight: 600; display: block; margin-bottom: 6px; }
        select, input[type=file] {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            margin-bottom: 14px;
        }
        button {
            width: 100%;
            border: none;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            color: #fff;
            background: #2563eb;
            cursor: pointer;
        }
        button:hover { background: #1d4ed8; }
        .result {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
        }
        .pred {
            font-size: 20px;
            font-weight: 600;
            margin: 0 0 12px;
        }
        .preview {
            width: 128px;
            height: 128px;
            image-rendering: pixelated;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            margin-bottom: 12px;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td {
            border-bottom: 1px solid #e5e7eb;
            padding: 8px 6px;
            text-align: left;
        }
        th { color: #6b7280; font-size: 13px; }
        .bar-wrap {
            height: 6px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }
        .bar { height: 100%; background: #3b82f6; }
        .error {
            margin-top: 14px;
            padding: 12px;
            border-radius: 8px;
            background: #fef2f2;
            color: #b91c1c;
        }
    </style>
</head>
<body>
<div class="wrap">
    <div class="card">
        <h1>CIFAR-100 分类</h1>

        <form id="predict-form" method="post" action="{{ url_for('predict_route') }}" enctype="multipart/form-data">
            <label for="task">任务</label>
            <select id="task" name="task">
                <option value="coarse" {% if selected_task == 'coarse' %}selected{% endif %}>粗标签 (20类)</option>
                <option value="fine" {% if selected_task == 'fine' %}selected{% endif %}>细标签 (100类)</option>
            </select>

            <label for="image">图片</label>
            <input id="image" type="file" name="image" accept="image/*" required>

            <button type="submit">预测</button>
        </form>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        {% if result %}
        <div class="result">
            <p class="pred">{{ result.class_name }} · {{ '%.1f' % (result.conf * 100) }}%</p>
            <img class="preview" src="data:image/png;base64,{{ preview }}" alt="preview">
            <table>
                <tr><th>类别</th><th>概率</th><th></th></tr>
                {% for item in result.top %}
                <tr>
                    <td>{{ item.name }}</td>
                    <td>{{ item.pct }}</td>
                    <td><div class="bar-wrap"><div class="bar" style="width: {{ '%.1f' % (item.prob * 100) }}%;"></div></div></td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
    </div>
</div>
</body>
</html>
"""


def _no_cache_response(html):
    response = make_response(html)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


def _render_page(**kwargs):
    defaults = {"selected_task": "coarse"}
    defaults.update(kwargs)
    return _no_cache_response(render_template_string(HTML, **defaults))


@app.route("/")
def index():
    return _render_page(selected_task=session.pop("selected_task", "coarse"))


@app.route("/result")
def show_result():
    payload = session.get("last_result")
    if not payload:
        return redirect(url_for("index"))
    return _render_page(**payload)


@app.route("/predict", methods=["POST"])
def predict_route():
    task = request.form.get("task", "coarse")
    file = request.files.get("image")
    if not file or not file.filename:
        session["selected_task"] = task
        return redirect(url_for("index"))

    try:
        raw_bytes = file.read()
        if not raw_bytes:
            raise ValueError("上传文件为空，请重新选择图片后再预测。")

        image = Image.open(io.BytesIO(raw_bytes))
        tensor, resized = preprocess(image)
        pred_class, conf, top_list, _ = predict(task, tensor, top_k=5)

        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        session["last_result"] = {
            "result": {
                "class_name": pred_class,
                "conf": conf,
                "top": top_list,
            },
            "preview": preview_b64,
            "selected_task": task,
        }
        return redirect(url_for("show_result"))
    except Exception as exc:
        session["selected_task"] = task
        return _render_page(error=str(exc), selected_task=task)


if __name__ == "__main__":
    coarse_path = find_latest_model("coarse")
    fine_path = find_latest_model("fine")
    print("=" * 50)
    print("CIFAR-100 PyTorch Web Demo")
    print(f"Device: {DEVICE}")
    print(f"Coarse model: {coarse_path or 'NOT FOUND'}")
    print(f"Fine model  : {fine_path or 'NOT FOUND'}")
    print("Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)