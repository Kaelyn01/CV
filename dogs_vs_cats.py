# %% [markdown]
# # 🐶🐱 Dogs vs Cats 二分类
# 数据集: Kaggle Dogs vs Cats (tongpython/cat-and-dog)
# 训练集 8007 张 | 测试集 2025 张

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# %%
# ========== 1. 数据路径 & 统计 ==========
DATA = Path("data/dogs-vs-cats")
train_dir = DATA / "training_set/training_set"
test_dir  = DATA / "test_set/test_set"

for split, path in [("训练集", train_dir), ("测试集", test_dir)]:
    cats = len(list((path / "cats").glob("*.jpg")))
    dogs = len(list((path / "dogs").glob("*.jpg")))
    print(f"{split}: 🐱 {cats} 张  🐶 {dogs} 张  (共 {cats+dogs})")

# %%
# ========== 2. 数据预处理 & 增强 ==========
# 训练: 随机翻转 + 归一化
# 测试: 只做归一化
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# %%
# ========== 3. 加载数据 ==========
full_train = datasets.ImageFolder(train_dir, transform=train_transform)
full_test  = datasets.ImageFolder(test_dir, transform=test_transform)
class_names = full_train.classes  # ['cats', 'dogs']
print(f"类别: {class_names} → 映射: {full_train.class_to_idx}")

# 从训练集切 20% 做验证
val_size = int(0.2 * len(full_train))
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(full_test,  batch_size=32, shuffle=False)

print(f"训练: {train_size} | 验证: {val_size} | 测试: {len(full_test)}")

# %%
# ========== 4. 看几张样本 ==========
# 用未归一化的 transform 来可视化
vis_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
vis_dataset = datasets.ImageFolder(train_dir, transform=vis_transform)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img, label = vis_dataset[i]
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(class_names[label])
    ax.axis("off")
plt.suptitle("训练集样本")
plt.tight_layout()
plt.show()

# %%
# ========== 5. 定义模型（迁移学习） ==========
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# 替换最后的全连接层为 2 分类
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
print(f"参数总数: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %%
# ========== 6. 训练配置 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 5

# %%
# ========== 7. 训练 & 验证循环 ==========
for epoch in range(epochs):
    # ---- 训练 ----
    model.train()
    train_loss = train_correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        train_correct += (model(imgs).argmax(1) == labels).sum().item()

    # ---- 验证 ----
    model.eval()
    val_loss = val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            val_loss += loss.item() * imgs.size(0)
            val_correct += (model(imgs).argmax(1) == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Acc: {train_correct/train_size:.3f} | "
          f"Val Acc: {val_correct/val_size:.3f}")

# %%
# ========== 8. 测试集评估 ==========
model.eval()
test_correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        test_correct += (model(imgs).argmax(1) == labels).sum().item()

print(f"测试集准确率: {test_correct/len(full_test):.3f}")

# %%
# ========== 9. 单张推理 ==========
import random
sample_img, true_label = full_test[random.randint(0, len(full_test)-1)]
model.eval()
with torch.no_grad():
    logits = model(sample_img.unsqueeze(0).to(device))
    pred = logits.argmax(1).item()
    prob = torch.softmax(logits, dim=1)[0][pred].item()

plt.imshow(sample_img.permute(1, 2, 0).cpu() * torch.tensor([0.229,0.224,0.225]) + torch.tensor([0.485,0.456,0.406]))
plt.title(f"真实: {class_names[true_label]} | 预测: {class_names[pred]} ({prob:.2%})")
plt.axis("off")
plt.show()
