from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
# 定义文件路径
img_path = "data/dogs-vs-cats/test_set/test_set/cats/cat.4821.jpg"
# 打开图像并启动open（）
img = Image.open(img_path)
# 将图像转换为张量
transform = transforms.ToTensor()
# 命名一个新变量来存储转换后的张量图像
tensor_img = transform(img)
print(tensor_img)  # Output: torch.Size([3, H, W]) where H and W are the height and width of the image