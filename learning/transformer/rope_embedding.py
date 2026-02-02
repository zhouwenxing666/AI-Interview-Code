import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) 的 PyTorch 实现。
    
    RoPE 通过将位置信息编码为旋转矩阵，并将其应用于 Query 和 Key 向量，
    从而实现相对位置编码。
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        """
        初始化 RoPE 模块。

        Args:
            dim (int): 嵌入维度 (通常是 head_dim)。
            max_seq_len (int): 最大序列长度。
            base (int): 计算 theta 的基数，默认为 10000。
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率 theta
        # theta_i = base ^ (-2i / d), 其中 i = 0, 1, ..., d/2 - 1
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 缓存 cos 和 sin 表
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, max_seq_len: int):
        """
        预计算并缓存 cos 和 sin 值。
        """
        self.max_seq_len = max_seq_len
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # freqs: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        
        # 将 freqs 扩展为 (seq_len, dim) 以匹配 query/key 的形状
        # emb: [theta_0, theta_1, ..., theta_{d/2-1}, theta_0, theta_1, ..., theta_{d/2-1}]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 注册为 buffer，不作为模型参数更新
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        获取对应序列长度的 cos 和 sin。

        Args:
            x (torch.Tensor): 输入张量。
            seq_len (int): 序列长度。

        Returns:
            cos, sin: 对应形状的 cos 和 sin 张量。
        """
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x: torch.Tensor):
    """
    将输入向量 x 分成两半，并进行旋转操作 [-x2, x1]。
    
    Args:
        x (torch.Tensor): 输入张量，形状为 (..., dim)。
        
    Returns:
        torch.Tensor: 旋转后的张量。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    将 RoPE 应用于 Query 和 Key。
    
    公式:
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    Args:
        q (torch.Tensor): Query 张量，形状 (batch_size, num_heads, seq_len, head_dim)。
        k (torch.Tensor): Key 张量，形状 (batch_size, num_heads, seq_len, head_dim)。
        cos (torch.Tensor): Cosine 张量。
        sin (torch.Tensor): Sine 张量。
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 旋转后的 q and k。
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 示例用法
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    num_heads = 4
    seq_len = 10
    head_dim = 64  # 必须是偶数
    
    # 初始化 RoPE
    rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=seq_len)
    
    # 模拟 Query 和 Key
    # 形状: (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"原始 Query 形状: {q.shape}")
    print(f"原始 Key 形状: {k.shape}")
    
    # 获取 cos 和 sin
    cos, sin = rope(q, seq_len)
    
    # 应用 RoPE
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"旋转后 Query 形状: {q_rot.shape}")
    print(f"旋转后 Key 形状: {k_rot.shape}")
    
    # 验证输出是否包含 NaN
    assert not torch.isnan(q_rot).any(), "Query 包含 NaN"
    assert not torch.isnan(k_rot).any(), "Key 包含 NaN"
    print("RoPE 应用成功！")
