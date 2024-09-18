import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
# 定义训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 计算损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

# 定义测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 计算损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    accuracy = correct / total
    print(f"Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    # 假设已经有 train_images_tensor 和 train_labels_tensor
    # 以及 test_images_tensor 和 test_labels_tensor

    # 加载保存的数据集
    train_dataset = torch.load('train_dataset.pt')
    test_dataset = torch.load('test_dataset.pt')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)




    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的 ResNet 模型，并进行微调
    model = models.resnet18(pretrained=True)

    # 替换最后一层，以适应当前任务（假设二分类任务）
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 二分类：‘Qualified’和‘Unqualified’

    # 将模型移到设备（GPU或CPU）
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数（用于分类任务）
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器


    # 开始训练
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}:")
        train(model, train_loader, criterion, optimizer, device)
        test(model, test_loader, criterion, device)
