# cv transform对图片进行变换
'''
常见的图像变换
1. 裁剪（Crop）
    中心裁剪：transforms.CenterCrop 
    随机裁剪：transforms.RandomCrop 
    随机长宽比裁剪：transforms.RandomResizedCrop 
    上下左右中心裁剪：transforms.FiveCrop 
    上下左右中心裁剪后翻转，transforms.TenCrop

2. 翻转和旋转（Flip and Rotation） 
    依概率p水平翻转：transforms.RandomHorizontalFlip(p=0.5) 
    依概率p垂直翻转：transforms.RandomVerticalFlip(p=0.5) 
    随机旋转：transforms.RandomRotation

3. 图像变换（resize）
    transforms.Resize 
    标准化：transforms.Normalize 
    转为tensor，并归一化至[0-1]：transforms.ToTensor 
    填充：transforms.Pad 
    修改亮度、对比度和饱和度：transforms.ColorJitter 
    转灰度图：transforms.Grayscale 
    线性变换：transforms.LinearTransformation() 
    仿射变换：transforms.RandomAffine 
    依概率p转为灰度图：transforms.RandomGrayscale 
    将数据转换为PILImage：transforms.ToPILImage transforms.Lambda：Apply a user-defined lambda as a transform.

4.对transforms操作，使数据增强更灵活 transforms.RandomChoice(transforms)， 
从给定的一系列transforms中选一个进行操作 transforms.RandomApply(transforms, p=0.5)，
给一个transform加上概率，依概率进行操作 transforms.RandomOrder，将transforms中的操作随机打乱
'''

'''
使用Google colab免费使用GPU运行
'''
