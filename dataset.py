import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir="./datasets", batch_size=128, num_workers=2):
    """
    加载 MNIST 数据集

    图像预处理：
    1. 转换为 Tensor
    2. 填充到 32×32
    3. 标准化
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("========== MNIST 数据集检查 ==========")
    print("训练集数量：", len(train_loader.dataset))
    print("测试集数量：", len(test_loader.dataset))
    print("图像尺寸：", images.shape)
    print("标签尺寸：", labels.shape)
    print("像素值范围：", images.min().item(), images.max().item())
    print("标签类别：", torch.unique(labels).tolist())
    print("数据加载测试成功！")
