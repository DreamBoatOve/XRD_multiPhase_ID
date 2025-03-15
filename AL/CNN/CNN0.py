"""
提供一个示例 PyTorch 代码，展示如何构建针对1D XRD数据输入、输出14 种 Bravais 晶格相含量（通过 Softmax 归一）的大体网络框架。
该代码包含从“小数据”到“大数据”的不同架构思路：最基础的浅层 CNN到可扩展的深度网络（可加入注意力或 Transformer/ViT/UNet 等）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# 1. 基础版 1D CNN 模型
# ----------------------
class XRDModelBase(nn.Module):
    """
    针对小数据(200~1000 条)的浅层 CNN 模型:
      - Input: (batch_size, 1, 4000)
      - Output: (batch_size, 14) + Softmax
    结构:
      Conv1D -> BN -> ReLU -> Pooling -> Flatten -> Dense -> Dense -> Softmax
    """
    def __init__(self, num_classes=14, init_channels=16, kernel_size=5, pool_size=2):
        super(XRDModelBase, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=init_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(init_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # 根据实际需要可再加一层卷积
        # self.conv2 = nn.Conv1d(init_channels, init_channels*2, kernel_size, stride=1, padding=kernel_size//2)
        # self.bn2 = nn.BatchNorm1d(init_channels*2)
        # self.pool2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # 全连接层
        # 由于经过 pool 后长度缩短，需要计算 flatten 后的特征数
        # 假设只使用 pool1, 原长度 4000 => 4000/pool_size => 2000 (若 pool_size=2)
        # 特征数 = out_channels * new_length
        self.fc1 = nn.Linear(init_channels * (4000 // pool_size), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: (batch_size, 1, 4000)
        """
        # 卷积 + BN + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # 池化
        x = self.pool1(x)  # (batch_size, init_channels, 4000//pool_size)

        # 若加了更多层，可在此继续卷积->BN->激活->池化

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, init_channels * (4000//pool_size))

        # 全连接
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # (batch_size, num_classes)

        # 输出前可做 softmax，但通常在 loss 中做
        # 这里仅演示: 直接返回 logits
        return x


# -----------------------
# 2. 深层可扩展 CNN (ResNet/Attention) 示例
# -----------------------
class BasicBlock(nn.Module):
    """
    简化版 ResNet block (1D)，可用于更深层 CNN
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class XRDModelResNet(nn.Module):
    """
    针对中/大规模数据的更深层 1D CNN:
      - 可堆叠多个 BasicBlock
      - 可在后续加 Attention, Transformer, etc.
    """
    def __init__(self, num_classes=14, layers=[16, 32, 64], kernel_size=3, pool_size=2):
        super(XRDModelResNet, self).__init__()
        self.in_channels = 1

        # 初始卷积
        self.conv1 = nn.Conv1d(self.in_channels, layers[0], kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(layers[0])

        # 残差层 (示例：2~3个 Block)
        self.res_blocks = nn.ModuleList()
        in_planes = layers[0]
        for out_planes in layers:
            self.res_blocks.append(BasicBlock(in_planes, out_planes, kernel_size=kernel_size, stride=1))
            in_planes = out_planes

        # 池化(可多次池化)
        self.pool = nn.AdaptiveMaxPool1d(output_size=4000//pool_size)

        # 全连接
        self.fc1 = nn.Linear(layers[-1]*(4000//pool_size), 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 初始卷积
        out = F.relu(self.bn1(self.conv1(x)))  # (batch_size, layers[0], seq_len)
        # 多个残差块
        for block in self.res_blocks:
            out = block(out)
        # 池化
        out = self.pool(out)  # (batch_size, layers[-1], 4000//pool_size)
        out = out.view(out.size(0), -1)  # flatten
        # 全连接
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# -----------------------
# 3. 训练示例
# -----------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device='cuda'):
    """
    训练单个 epoch 的示例函数
    model: 神经网络模型
    dataloader: PyTorch DataLoader, 提供 (X, y) batch
    optimizer: 优化器
    criterion: 损失函数(可用CrossEntropy/KLDivLoss等)
    device: 'cuda' 或 'cpu'
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        # X_batch shape: (batch_size, 1, 4000), y_batch shape: (batch_size, 14)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        # 前向
        logits = model(X_batch)  # (batch_size, 14)
        # 这里假设 y_batch 也是概率分布 (batch_size, 14)，可用 KLDivLoss
        # 或若 y_batch 为类别, 则用 CrossEntropyLoss
        loss = criterion(logits, y_batch)
        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def main():
    # 假设我们有 2000 条训练数据, batch_size=32
    # X shape: (2000, 1, 4000), y shape: (2000, 14)
    # 这里只是示例, 需要你自行构建 dataset/dataloader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = XRDModelBase(num_classes=14, init_channels=16, kernel_size=5, pool_size=2).to(device)

    # 示例: CrossEntropyLoss 适合分类, 若是输出相含量概率, 可考虑 KLDivLoss + log_softmax
    # 假设 y 是 one-hot 或 soft label, 这里先用 MSELoss 也可以(不推荐), 仅演示
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 伪代码：dataloader = ...
    # for epoch in range(10):
    #     train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
    #     print(f"Epoch {epoch}, Loss: {train_loss:.4f}")

    print("模型结构:")
    print(model)
    print("请自行实现 DataLoader 并进行训练。")


if __name__ == "__main__":
    main()
