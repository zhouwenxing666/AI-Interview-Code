"""
交叉熵损失函数PyTorch实现
Cross-Entropy Loss Function Implementation in PyTorch

作者: AI Assistant
日期: 2024
功能: 提供多种交叉熵损失函数的实现，包括标准交叉熵、带权重的交叉熵、标签平滑等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class CrossEntropyLoss(nn.Module):
    """
    标准交叉熵损失函数实现
    
    交叉熵损失函数是分类任务中最常用的损失函数之一，
    它衡量预测概率分布与真实标签分布之间的差异。
    
    数学公式: CE = -∑(y_i * log(p_i))
    其中 y_i 是真实标签的one-hot编码，p_i 是预测概率
    """
    
    def __init__(self, 
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        """
        初始化交叉熵损失函数
        
        Args:
            weight (Optional[torch.Tensor]): 类别权重，用于处理类别不平衡问题
            reduction (str): 损失缩减方式，可选 'mean', 'sum', 'none'
            label_smoothing (float): 标签平滑参数，范围 [0, 1)
        """
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算交叉熵损失
        
        Args:
            input (torch.Tensor): 模型预测的logits，形状为 (N, C) 或 (N, C, ...)
            target (torch.Tensor): 真实标签，形状为 (N,) 或 (N, ...)
            
        Returns:
            torch.Tensor: 计算得到的交叉熵损失
        """
        return F.cross_entropy(input, target, 
                              weight=self.weight, 
                              reduction=self.reduction,
                              label_smoothing=self.label_smoothing)


class CustomCrossEntropyLoss(nn.Module):
    """
    自定义交叉熵损失函数实现
    
    从零开始实现交叉熵损失函数，用于深入理解其数学原理
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        初始化自定义交叉熵损失函数
        
        Args:
            reduction (str): 损失缩减方式
        """
        super(CustomCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        自定义交叉熵损失函数的前向传播
        
        Args:
            input (torch.Tensor): 模型预测的logits
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 计算得到的交叉熵损失
        """
        # 应用softmax得到概率分布
        probabilities = F.softmax(input, dim=1)
        
        # 添加数值稳定性处理，避免log(0)
        probabilities = torch.clamp(probabilities, min=1e-8, max=1.0)
        
        # 计算负对数似然
        log_probabilities = torch.log(probabilities)
        
        # 获取目标类别的对数概率
        batch_size = input.size(0)
        target_log_probs = log_probabilities[range(batch_size), target]
        
        # 计算损失
        loss = -target_log_probs
        
        # 根据reduction参数处理损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    带权重的交叉熵损失函数
    
    用于处理类别不平衡问题，给少数类别更高的权重
    """
    
    def __init__(self, 
                 class_weights: Union[List[float], torch.Tensor],
                 reduction: str = 'mean'):
        """
        初始化带权重的交叉熵损失函数
        
        Args:
            class_weights (Union[List[float], torch.Tensor]): 各类别的权重
            reduction (str): 损失缩减方式
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        if isinstance(class_weights, list):
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        带权重的交叉熵损失函数前向传播
        
        Args:
            input (torch.Tensor): 模型预测的logits
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 计算得到的加权交叉熵损失
        """
        # 将权重移到与input相同的设备
        weights = self.class_weights.to(input.device)
        
        # 计算标准交叉熵损失
        loss = F.cross_entropy(input, target, reduction='none')
        
        # 应用权重
        weighted_loss = loss * weights[target]
        
        # 根据reduction参数处理损失
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:  # 'none'
            return weighted_loss


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    
    Focal Loss是专门为处理类别不平衡问题而设计的损失函数，
    通过降低易分类样本的权重来关注难分类样本。
    
    数学公式: FL = -α(1-p_t)^γ * log(p_t)
    其中 p_t 是预测概率，α 是权重因子，γ 是聚焦参数
    """
    
    def __init__(self, 
                 alpha: Union[float, List[float]] = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha (Union[float, List[float]]): 权重因子
            gamma (float): 聚焦参数，控制难易样本的权重差异
            reduction (str): 损失缩减方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss前向传播
        
        Args:
            input (torch.Tensor): 模型预测的logits
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 计算得到的Focal Loss
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(input, target, reduction='none')
        
        # 计算预测概率
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 根据reduction参数处理损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def calculate_class_weights(labels: torch.Tensor, 
                           num_classes: int,
                           method: str = 'balanced') -> torch.Tensor:
    """
    计算类别权重，用于处理类别不平衡问题
    
    Args:
        labels (torch.Tensor): 真实标签
        num_classes (int): 类别数量
        method (str): 权重计算方法，可选 'balanced', 'inverse'
        
    Returns:
        torch.Tensor: 计算得到的类别权重
    """
    # 统计每个类别的样本数量
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    
    if method == 'balanced':
        # 平衡权重：总样本数 / (类别数 * 各类别样本数)
        total_samples = len(labels)
        weights = total_samples / (num_classes * class_counts)
        # 避免除零错误
        weights = torch.where(class_counts > 0, weights, 0.0)
    elif method == 'inverse':
        # 反比权重：1 / 各类别样本数
        weights = 1.0 / (class_counts + 1e-8)  # 添加小常数避免除零
    else:
        raise ValueError(f"不支持的权重计算方法: {method}")
    
    return weights


def test_cross_entropy_losses():
    """
    测试各种交叉熵损失函数的实现
    
    创建测试数据并验证各种损失函数的计算结果
    """
    print("=" * 60)
    print("交叉熵损失函数测试")
    print("=" * 60)
    
    # 设置随机种子确保结果可重现
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 4
    num_classes = 3
    
    # 模拟模型预测的logits
    logits = torch.randn(batch_size, num_classes)
    print(f"输入logits形状: {logits.shape}")
    print(f"输入logits:\n{logits}")
    
    # 模拟真实标签
    targets = torch.tensor([0, 1, 2, 0])
    print(f"真实标签: {targets}")
    
    # 1. 标准交叉熵损失
    print("\n1. 标准交叉熵损失:")
    ce_loss = CrossEntropyLoss()
    loss1 = ce_loss(logits, targets)
    print(f"标准交叉熵损失: {loss1.item():.4f}")
    
    # 2. 自定义交叉熵损失
    print("\n2. 自定义交叉熵损失:")
    custom_ce_loss = CustomCrossEntropyLoss()
    loss2 = custom_ce_loss(logits, targets)
    print(f"自定义交叉熵损失: {loss2.item():.4f}")
    
    # 3. 带权重的交叉熵损失
    print("\n3. 带权重的交叉熵损失:")
    class_weights = [1.0, 2.0, 1.5]  # 给类别1更高权重
    weighted_ce_loss = WeightedCrossEntropyLoss(class_weights)
    loss3 = weighted_ce_loss(logits, targets)
    print(f"带权重的交叉熵损失: {loss3.item():.4f}")
    
    # 4. Focal Loss
    print("\n4. Focal Loss:")
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    loss4 = focal_loss(logits, targets)
    print(f"Focal Loss: {loss4.item():.4f}")
    
    # 5. 标签平滑交叉熵损失
    print("\n5. 标签平滑交叉熵损失:")
    smooth_ce_loss = CrossEntropyLoss(label_smoothing=0.1)
    loss5 = smooth_ce_loss(logits, targets)
    print(f"标签平滑交叉熵损失: {loss5.item():.4f}")
    
    # 6. 计算类别权重
    print("\n6. 类别权重计算:")
    weights = calculate_class_weights(targets, num_classes, method='balanced')
    print(f"平衡权重: {weights}")
    
    # 验证与PyTorch内置函数的对比
    print("\n7. 与PyTorch内置函数对比:")
    pytorch_loss = F.cross_entropy(logits, targets)
    print(f"PyTorch内置交叉熵损失: {pytorch_loss.item():.4f}")
    print(f"自定义实现与PyTorch差异: {abs(loss2.item() - pytorch_loss.item()):.6f}")


def demonstrate_usage():
    """
    演示交叉熵损失函数的使用方法
    
    展示在实际项目中如何使用这些损失函数
    """
    print("\n" + "=" * 60)
    print("交叉熵损失函数使用演示")
    print("=" * 60)
    
    # 模拟一个简单的分类任务
    torch.manual_seed(123)
    
    # 创建模拟数据
    batch_size = 8
    num_classes = 5
    input_size = 10
    
    # 模拟一个简单的神经网络
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 创建模型和数据
    model = SimpleClassifier(input_size, num_classes)
    x = torch.randn(batch_size, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # 前向传播
    logits = model(x)
    
    print(f"模型输出logits形状: {logits.shape}")
    print(f"真实标签: {y}")
    
    # 使用不同的损失函数
    losses = {}
    
    # 标准交叉熵
    ce_loss = CrossEntropyLoss()
    losses['标准交叉熵'] = ce_loss(logits, y)
    
    # 带权重的交叉熵（模拟类别不平衡）
    weights = [1.0, 3.0, 1.0, 2.0, 1.0]  # 类别1权重更高
    weighted_ce = WeightedCrossEntropyLoss(weights)
    losses['带权重交叉熵'] = weighted_ce(logits, y)
    
    # Focal Loss
    focal = FocalLoss(alpha=1.0, gamma=2.0)
    losses['Focal Loss'] = focal(logits, y)
    
    # 标签平滑
    smooth_ce = CrossEntropyLoss(label_smoothing=0.1)
    losses['标签平滑'] = smooth_ce(logits, y)
    
    print("\n各种损失函数结果:")
    for name, loss in losses.items():
        print(f"{name}: {loss.item():.4f}")
    
    # 演示梯度计算
    print("\n梯度计算演示:")
    loss = ce_loss(logits, y)
    loss.backward()
    
    # 检查模型参数的梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} 梯度范数: {param.grad.norm().item():.4f}")


if __name__ == "__main__":
    """
    主函数：运行所有测试和演示
    """
    print("PyTorch交叉熵损失函数实现")
    print("作者: AI Assistant")
    print("功能: 提供多种交叉熵损失函数的完整实现")
    
    # 运行测试
    test_cross_entropy_losses()
    
    # 运行使用演示
    demonstrate_usage()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
