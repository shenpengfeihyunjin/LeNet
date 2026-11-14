import os
import csv
import torch
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from torchvision.utils import make_grid
from dataset import get_dataloaders
from models.lenet import LeNet5


DEVICE = torch.device("cuda:1")
CHECKPOINT = "outputs/checkpoints/best_lenet.pth"
OUTPUT_DIR = "outputs"
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def main():
    print("========== LeNet 最佳模型测试集评估 ==========")
    print("评估设备：", DEVICE)

    _, test_loader = get_dataloaders(batch_size=128)

    model = LeNet5().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    sample_images = []
    sample_labels = []
    sample_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            if len(sample_images) < 40:
                remaining = 40 - len(sample_images)
                sample_images.extend(images[:remaining].cpu())
                sample_labels.extend(labels[:remaining].cpu().tolist())
                sample_predictions.extend(predictions[:remaining].cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    test_loss = total_loss / len(test_loader.dataset)
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(10)),
        target_names=[str(i) for i in range(10)],
        digits=4,
        zero_division=0
    )

    print(f"最佳模型轮次：{checkpoint['epoch']}")
    print(f"测试集数量：{len(y_true)}")
    print(f"测试集损失：{test_loss:.6f}")
    print(f"测试集准确率：{test_accuracy:.2f}%")
    print("\n分类指标：")
    print(report)

    # 保存测试结果
    with open(os.path.join(OUTPUT_DIR, "test_results.txt"),
              "w", encoding="utf-8") as f:
        f.write("LeNet 最佳模型测试集评估结果\n")
        f.write(f"最佳模型轮次：{checkpoint['epoch']}\n")
        f.write(f"测试集数量：{len(y_true)}\n")
        f.write(f"测试集损失：{test_loss:.6f}\n")
        f.write(f"测试集准确率：{test_accuracy:.2f}%\n\n")
        f.write(report)

    # 保存混淆矩阵
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("LeNet Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(10), range(10))
    plt.yticks(range(10), range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "lenet_confusion_matrix.png"),
        dpi=300
    )
    plt.close()

    # 保存混淆矩阵数值
    with open(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
              "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["True/Pred"] + list(range(10)))
        for i, row in enumerate(cm):
            writer.writerow([i] + row.tolist())

    # 绘制40张预测示例
    images = torch.stack(sample_images)
    grid = make_grid(images, nrow=8, normalize=True, padding=2)

    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.title("LeNet MNIST Prediction Examples")

    for idx, (label, pred) in enumerate(
        zip(sample_labels, sample_predictions)
    ):
        row, col = divmod(idx, 8)
        plt.text(
            col * 1.25 + 0.2,
            row * 1.25 + 0.15,
            f"T:{label} P:{pred}",
            color="red" if label != pred else "green",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(PREDICTION_DIR, "lenet_prediction_examples.png"),
        dpi=300
    )
    plt.close()

    print("\n========== 评估完成 ==========")
    print("测试结果：outputs/test_results.txt")
    print("混淆矩阵：outputs/confusion_matrix.csv")
    print("混淆矩阵图：outputs/plots/lenet_confusion_matrix.png")
    print("预测示例：outputs/predictions/lenet_prediction_examples.png")


if __name__ == "__main__":
    main()
