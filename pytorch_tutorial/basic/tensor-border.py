from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# writer.add_image('four_fashion_mnist_images', img_grid) # 展示图片

# tensorboard --logdir=runs --port 8123

# writer.add_graph(net, images) #查看模型结构

# writer.add_embedding #高维数据的低维度表示
