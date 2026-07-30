from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
img = Image.open("data/dogs-vs-cats/test_set/test_set/cats/cat.4909.jpg")
print(img)

trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)
writer.add_image('cat', img_tensor)
writer.close()
