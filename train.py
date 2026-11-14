import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.lenet import LeNet5
from dataset import get_dataloaders


# ==================== 实验配置 ====================
GPU_ID = 1
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_WORKERS = 2

DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("========== LeNet 训练开始 ==========")
    print(f"设备：{DEVICE}")
    print(f"训练轮数：{EPOCHS}")
    print(f"Batch Size：{BATCH_SIZE}")
    print(f"学习率：{LEARNING_RATE}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，已停止训练，没有自动切换到 CPU。")

    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = LeNet5().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0
    history = []

    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

        train_loss = running_loss / total
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_accuracy": test_accuracy
            }, os.path.join(CHECKPOINT_DIR, "best_lenet.pth"))
            print(f"  已保存最佳模型，准确率：{best_accuracy:.2f}%")

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "test_loss", "test_accuracy"]
            )
            writer.writeheader()
            writer.writerows(history)

    # 保存最后一轮模型
    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "last_lenet.pth")
    )

    # 绘制损失曲线
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    test_losses = [item["test_loss"] for item in history]
    accuracies = [item["test_accuracy"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LeNet Training and Test Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lenet_loss_curve.png"), dpi=300)
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("LeNet Test Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lenet_accuracy_curve.png"), dpi=300)
    plt.close()

    print("\n========== 训练完成 ==========")
    print(f"最佳测试准确率：{best_accuracy:.2f}%")
    print(f"最佳模型：{CHECKPOINT_DIR}/best_lenet.pth")
    print(f"训练日志：{log_path}")
    print(f"损失曲线：{PLOT_DIR}/lenet_loss_curve.png")
    print(f"准确率曲线：{PLOT_DIR}/lenet_accuracy_curve.png")


if __name__ == "__main__":
    main()
