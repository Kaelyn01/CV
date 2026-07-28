# %% [markdown]
# # 交互式 PyTorch 沙盒
# 用法和 PyCharm 控制台一样：
# - 每个 `# %%` 是一个可独立执行的 cell
# - 🔘 点击 **Run Cell** 或按 **Shift+Enter** 执行
# - 🧹 变量在所有 cell 间持久化，就像控制台
# - 📊 右键变量可以查看数据 (View variable in data viewer)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

print(f"✅ PyTorch {torch.__version__}")
print(f"✅ MPS (Apple GPU) 可用: {torch.backends.mps.is_available()}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# %%
# ========== 随便试试 ==========
# 创建一个张量，随时可以改、可以重新跑这个 cell
x = torch.randn(3, 4)
print(f"shape: {x.shape}\ndtype: {x.dtype}\ndevice: {x.device}")
x

# %%
# ========== 自动梯度 ==========
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (w ** 2).sum()
y.backward()
print(f"w.grad = {w.grad}")  # dy/dw = 2w

# %%
# ========== 快速搭一个网络试试 ==========
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

model = TinyNet().to(device)
dummy_input = torch.randn(4, 10).to(device)
output = model(dummy_input)
print(f"输入: {dummy_input.shape} → 输出: {output.shape}")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# %%
# ========== 查看变量 ==========
# VS Code 左边的 Jupyter 面板可以看所有当前变量及其值
# 就像 PyCharm 的 Variables 面板一样
a = torch.arange(0, 10)
b = a * 2 + torch.randn(10) * 0.5
print(f"a = {a.tolist()}")
print(f"b = {b.tolist()}")

# %%
# 想清空所有变量？点 Interactive Window 顶部的 Restart 按钮
# 想执行当前 cell 及之后所有 cell？用 Cmd+Shift+Enter
