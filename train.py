

import os
import json
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np

from data_loader import get_dataloaders
from models import get_model


# 固定随机种子，开启 cuDNN 自动选最快卷积算法
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


# 训练一个 epoch：前向传播、算 loss、反向传播、更新权重（支持 AMP）
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, acc


# 把类别名转成安全文件名（去掉特殊字符）
def _safe_filename(name):
    return "".join(c if c.isalnum() or c in '-_' else '_' for c in str(name))


def save_tensor_image(tensor_chw, path):
    """Save CHW image tensor. Training data uses pixel range [0, 1]."""
    img = tensor_chw.detach().cpu().float()
    if img.max() > 1.0:
        img = img.clamp(0, 255)
    else:
        img = (img.clamp(0, 1) * 255.0)
    arr = img.permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(arr).save(path)


# 在测试集上评估：准确率/精确率/召回率/F1、混淆矩阵，并记录错分样本
def evaluate(model, loader, criterion, device, class_names, error_save_dir=None, max_errors=20):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    error_samples = []
    total_errors = 0
    sample_offset = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            wrong_mask = preds != labels
            if wrong_mask.any():
                total_errors += int(wrong_mask.sum().item())
                for i in range(labels.size(0)):
                    if wrong_mask[i] and len(error_samples) < max_errors:
                        pred_idx = preds[i].item()
                        true_idx = labels[i].item()
                        pred_prob = float(probs[i, pred_idx].item())
                        true_prob = float(probs[i, true_idx].item())
                        top_probs, top_idx = torch.topk(probs[i], k=min(5, probs.size(1)))
                        top_predictions = [
                            {
                                'rank': rank + 1,
                                'label_id': int(idx.item()),
                                'class': class_names[int(idx.item())],
                                'prob': float(prob.item()),
                                'pct': f"{float(prob.item()) * 100:.2f}%",
                            }
                            for rank, (prob, idx) in enumerate(zip(top_probs, top_idx))
                        ]
                        entry = {
                            'index': sample_offset + i,
                            'true_label': true_idx,
                            'pred_label': pred_idx,
                            'true_class': class_names[true_idx],
                            'pred_class': class_names[pred_idx],
                            'pred_confidence': pred_prob,
                            'pred_confidence_pct': f"{pred_prob * 100:.2f}%",
                            'true_class_probability': true_prob,
                            'true_class_probability_pct': f"{true_prob * 100:.2f}%",
                            'top5_predictions': top_predictions,
                        }
                        if error_save_dir is not None:
                            os.makedirs(error_save_dir, exist_ok=True)
                            safe_true = _safe_filename(entry['true_class'])
                            safe_pred = _safe_filename(entry['pred_class'])
                            img_path = os.path.join(
                                error_save_dir,
                                f"err_{len(error_samples):02d}_idx{entry['index']}_"
                                f"true-{safe_true}_pred-{safe_pred}.png"
                            )
                            save_tensor_image(inputs[i], img_path)
                            entry['image_path'] = img_path
                        error_samples.append(entry)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            sample_offset += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': epoch_loss,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'report': report,
        'cm': cm,
        'preds': all_preds,
        'labels': all_labels,
        'error_samples': error_samples,
        'total_errors': total_errors,
    }


# 读取已有训练日志（续训时合并历史曲线）
def load_training_history(run_dir):
    log_path = os.path.join(run_dir, 'training_log.csv')
    if not os.path.exists(log_path):
        return []

    float_fields = {'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1'}
    with open(log_path, 'r', newline='', encoding='utf-8') as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = {'epoch': int(row['epoch'])}
            for key in float_fields:
                parsed[key] = float(row[key])
            rows.append(parsed)
        return rows


# 保存每轮训练指标到 CSV（供画 loss 曲线）
def save_training_log(run_dir, history):
    log_path = os.path.join(run_dir, 'training_log.csv')
    fieldnames = [
        'epoch', 'train_loss', 'train_acc',
        'val_loss', 'val_acc', 'val_prec', 'val_rec', 'val_f1',
    ]
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training log: {log_path}")
    return log_path


# 绘制训练/验证的 loss 和 accuracy 曲线
def plot_training_history(run_dir, history, label_type):
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [row['train_loss'] for row in history], label='train', marker='o', markersize=3)
    axes[0].plot(epochs, [row['val_loss'] for row in history], label='val', marker='o', markersize=3)
    axes[0].set_title(f'{label_type} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [row['train_acc'] for row in history], label='train', marker='o', markersize=3)
    axes[1].plot(epochs, [row['val_acc'] for row in history], label='val', marker='o', markersize=3)
    axes[1].set_title(f'{label_type} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, 'training_history.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot: {out_path}")
    return out_path


# 绘制混淆矩阵热力图
def plot_confusion_matrix(run_dir, cm, class_names, label_type):
    num_classes = len(class_names)
    fig_size = max(8, min(24, num_classes * 0.22))
    tick_font = 8 if num_classes <= 30 else 4

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        cmap='Blues',
        square=True,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f'{label_type} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=90, fontsize=tick_font)
    plt.yticks(rotation=0, fontsize=tick_font)
    plt.tight_layout()

    out_path = os.path.join(run_dir, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {out_path}")
    return out_path


# 汇总保存课设报告素材：metrics.json、分类报告、曲线图、混淆矩阵
def save_run_artifacts(run_dir, val_metrics, args, best_path, history=None):
    os.makedirs(run_dir, exist_ok=True)

    metrics = {
        'label_type': args.label,
        'model': args.model,
        'checkpoint': os.path.abspath(best_path),
        'accuracy': val_metrics['acc'],
        'precision_macro': val_metrics['prec'],
        'recall_macro': val_metrics['rec'],
        'f1_macro': val_metrics['f1'],
        'val_loss': val_metrics['loss'],
        'total_errors': val_metrics['total_errors'],
    }
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics: {metrics_path}")

    report_path = os.path.join(run_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(val_metrics['report'])
    print(f"Saved classification report: {report_path}")

    plot_confusion_matrix(run_dir, val_metrics['cm'], val_metrics.get('class_names_display', []), args.label)

    if history:
        save_training_log(run_dir, history)
        plot_training_history(run_dir, history, args.label)

    return metrics_path, report_path


# 从模型文件名解析实验目录名
def run_name_from_checkpoint(checkpoint_path):
    return os.path.basename(checkpoint_path).replace('_best.pth', '')


# 加载最佳模型，做最终评估，导出错题本和全部 outputs 文件
def finalize_run(model, test_loader, criterion, device, class_names, args, best_path, run_dir, history=None):
    errors_dir = os.path.join(run_dir, 'errors')
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    val_metrics = evaluate(
        model, test_loader, criterion, device, class_names,
        error_save_dir=errors_dir, max_errors=20
    )
    val_metrics['class_names_display'] = class_names

    error_json_path = os.path.join(run_dir, 'error_samples.json')
    with open(error_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'label_type': args.label,
            'model': args.model,
            'total_errors': val_metrics['total_errors'],
            'saved_samples': len(val_metrics['error_samples']),
            'fields': [
                'index', 'true_class', 'pred_class',
                'pred_confidence', 'true_class_probability', 'top5_predictions', 'image_path',
            ],
            'error_samples': val_metrics['error_samples'],
        }, f, ensure_ascii=False, indent=2)

    error_csv_path = os.path.join(run_dir, 'error_samples.csv')
    with open(error_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'index', 'true_class', 'pred_class',
                'pred_confidence_pct', 'true_class_probability_pct',
                'top1', 'top2', 'top3', 'top4', 'top5', 'image_path',
            ],
        )
        writer.writeheader()
        for err in val_metrics['error_samples']:
            tops = err.get('top5_predictions', [])
            writer.writerow({
                'index': err['index'],
                'true_class': err['true_class'],
                'pred_class': err['pred_class'],
                'pred_confidence_pct': err.get('pred_confidence_pct', ''),
                'true_class_probability_pct': err.get('true_class_probability_pct', ''),
                'top1': f"{tops[0]['class']} ({tops[0]['pct']})" if len(tops) > 0 else '',
                'top2': f"{tops[1]['class']} ({tops[1]['pct']})" if len(tops) > 1 else '',
                'top3': f"{tops[2]['class']} ({tops[2]['pct']})" if len(tops) > 2 else '',
                'top4': f"{tops[3]['class']} ({tops[3]['pct']})" if len(tops) > 3 else '',
                'top5': f"{tops[4]['class']} ({tops[4]['pct']})" if len(tops) > 4 else '',
                'image_path': err.get('image_path', ''),
            })
    print(f"Saved error CSV: {error_csv_path}")

    save_run_artifacts(run_dir, val_metrics, args, best_path, history=history)

    print("\n=== Final Test Report (best model) ===")
    print(f"Accuracy : {val_metrics['acc']:.4f}")
    print(f"Macro F1 : {val_metrics['f1']:.4f}")
    print(f"Total misclassifications: {val_metrics['total_errors']}")
    print("\nClassification Report (first 20 lines):")
    print('\n'.join(val_metrics['report'].splitlines()[:20]))

    print(f"\n=== Misclassification Notebook (first {len(val_metrics['error_samples'])} samples) ===")
    for err in val_metrics['error_samples']:
        print(f"  #{err['index']:5d}: true={err['true_class']} -> pred={err['pred_class']}")
    print(f"Run artifacts saved to: {run_dir}")
    print(f"Error manifest: {error_json_path}")
    print(f"Error table: {error_csv_path}")
    return val_metrics


def main():
    # 解析命令行：粗/细标签、模型类型、epoch 数等
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', choices=['coarse', 'fine'], default='coarse')
    parser.add_argument('--model', choices=['resnet'], default='resnet')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default='cifar-100-python')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Skip training and regenerate reports/plots from an existing *_best.pth',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from an existing *_best.pth for additional --epochs',
    )
    args = parser.parse_args()

    if args.checkpoint and args.resume:
        parser.error('Use only one of --checkpoint or --resume')

    # 固定随机种子，选择 GPU 或 CPU
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 仅评估模式：不训练，用已有 checkpoint 重新生成报告和错题
    if args.checkpoint:
        best_path = args.checkpoint
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        args.label = ckpt.get('label_type', args.label)
        class_names = ckpt['class_names']
        num_classes = ckpt.get('num_classes', len(class_names))
        run_name = run_name_from_checkpoint(best_path)
        run_dir = os.path.join('outputs', run_name)

        train_loader, test_loader, _, _ = get_dataloaders(
            args.data_dir, label_type=args.label, batch_size=args.batch_size
        )
        model = get_model(args.model, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        print(f"Eval-only mode: {best_path}")
        print(f"Output dir: {run_dir}")
        finalize_run(model, test_loader, criterion, device, class_names, args, best_path, run_dir)
        return

    # 加载 CIFAR-100 数据集（train/test）和模型
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        ckpt = torch.load(args.resume, map_location=device)
        args.label = ckpt.get('label_type', args.label)
        class_names = ckpt['class_names']
        num_classes = ckpt.get('num_classes', len(class_names))

        train_loader, test_loader, _, _ = get_dataloaders(
            args.data_dir, label_type=args.label, batch_size=args.batch_size
        )
        model = get_model(args.model, num_classes).to(device)
        model.load_state_dict(ckpt['model_state'])

        base_name = run_name_from_checkpoint(args.resume)
        prev_run_dir = os.path.join('outputs', base_name)
        history = load_training_history(prev_run_dir)
        start_epoch = len(history)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{base_name}_resume_{timestamp}"
        save_dir = 'models'
        run_dir = os.path.join('outputs', run_name)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        best_path = os.path.join(save_dir, f"{run_name}_best.pth")

        print(f"Resume from: {args.resume}")
        print(f"Previous epochs: {start_epoch}")
        print(f"Additional epochs: {args.epochs}")
        print(f"Run output dir: {run_dir}")
    else:
        train_loader, test_loader, num_classes, class_names = get_dataloaders(
            args.data_dir, label_type=args.label, batch_size=args.batch_size
        )
        model = get_model(args.model, num_classes).to(device)

        start_epoch = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.label}_{args.model}_{timestamp}"
        save_dir = 'models'
        run_dir = os.path.join('outputs', run_name)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        best_path = os.path.join(save_dir, f"{run_name}_best.pth")
        history = []

    print(f"Model: {args.model} | Classes: {num_classes}")

    # 定义损失函数、优化器、学习率调度、混合精度 scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    if args.resume:
        val_metrics = evaluate(model, test_loader, criterion, device, class_names)
        best_acc = val_metrics['acc']
        torch.save({
            'model_state': model.state_dict(),
            'num_classes': num_classes,
            'class_names': class_names,
            'label_type': args.label,
        }, best_path)
        print(f"Resume baseline val acc: {best_acc:.4f}")
        print(f"Saved resume checkpoint: {best_path}")
    else:
        best_acc = 0.0

    end_epoch = start_epoch + args.epochs
    print(f"\nStarting training for epochs {start_epoch + 1} to {end_epoch}...")

    # 训练循环：每轮训练后在测试集验证，保存 val acc 最高的权重
    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_metrics = evaluate(model, test_loader, criterion, device, class_names)

        scheduler.step(val_metrics['loss'])
        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_metrics['loss'], 6),
            'val_acc': round(val_metrics['acc'], 6),
            'val_prec': round(val_metrics['prec'], 6),
            'val_rec': round(val_metrics['rec'], 6),
            'val_f1': round(val_metrics['f1'], 6),
        })

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
              f"F1: {val_metrics['f1']:.4f}")

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            torch.save({
                'model_state': model.state_dict(),
                'num_classes': num_classes,
                'class_names': class_names,
                'label_type': args.label
            }, best_path)
            print(f"  Saved best model (val acc {best_acc:.4f})")

    print(f"\nTraining finished. Best val acc: {best_acc:.4f}")
    print(f"Best model saved to: {best_path}")

    # 用最佳模型做最终评估，生成错题本、图表和报告文件
    finalize_run(model, test_loader, criterion, device, class_names, args, best_path, run_dir, history=history)


if __name__ == "__main__":
    main()