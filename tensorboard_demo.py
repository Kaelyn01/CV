from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
for i in range(100):
    writer.add_scalar('Loss/train', 0.1 * i, i)
    writer.add_scalar('Loss/test', 0.1 * i + 0.5, i)
writer.close()
print("TensorBoard demo completed. You can now run 'tensorboard --logdir=runs' to visualize the results.")