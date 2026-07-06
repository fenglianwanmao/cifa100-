
import torch
from PIL import Image
import numpy as np
import os
import pickle

from models import get_model


def load_class_names(data_dir='cifar-100-python'):
    meta = pickle.load(open(os.path.join(data_dir, 'meta'), 'rb'), encoding='latin1')
    return meta['coarse_label_names'], meta['fine_label_names']


@torch.no_grad()
def predict_image(image_path, model_path, task='coarse', data_dir='../cifar-100-python', top_k=5):
    """
    Predict on a single image file.
    Returns: (predicted_class_name, confidence, list of topk dicts)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    num_classes = ckpt.get('num_classes', 100 if task == 'fine' else 20)
    class_names = ckpt.get('class_names')

    if class_names is None:
        coarse_names, fine_names = load_class_names(data_dir)
        class_names = fine_names if task == 'fine' else coarse_names

    model = get_model('resnet', num_classes).to(device)  # 默认用 resnet，和训练一致
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Preprocess
    img = Image.open(image_path).convert('RGB').resize((32, 32))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # Predict
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    conf, pred_idx = torch.max(probs, 0)

    top_probs, top_idx = torch.topk(probs, top_k)
    top_results = [
        {'name': class_names[i.item()], 'prob': float(p), 'pct': f"{float(p)*100:.2f}%"}
        for p, i in zip(top_probs, top_idx)
    ]

    return class_names[pred_idx.item()], float(conf), top_results


if __name__ == "__main__":
    # Example
    print("Inference module ready. Use from web app or scripts.")
    # pred, conf, tops = predict_image("some.png", "models/coarse_..._best.pth", task='coarse')
    # print(pred, conf)