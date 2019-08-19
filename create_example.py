import numpy as np
import os
import PIL.Image as pimg

'''该类用于生成训练与测试数据集'''


# 用于创建大小为224*224的背景图片
def createBgPhoto(bg_path, save_path):
    index = 0
    bg_photos = os.listdir(bg_path)
    for photo in bg_photos:
        bg_photo = pimg.open(os.path.join(bg_path, photo))
        bg_convert = bg_photo.convert("RGB")
        bg_resize = bg_convert.resize((224, 224))
        bg_resize.save("{0}/{1}.png".format(save_path, str(index) + ".0.0.0.0.0"))
        index += 1


# 用于将小黄人粘贴到背景图片上
def pasteMinions(bg_path, minions_path, save_path):
    index = 0
    bg_photos = os.listdir(bg_path)
    for photo in bg_photos:
        # 随机粘贴第几个小黄人的图片
        minios_random = np.random.randint(1, 21)
        # 小黄人图片的随机宽高
        width_random = np.random.randint(50, 100)
        height_random = np.random.randint(50, 100)
        # 小黄人图片随机的旋转角度
        rotate_random = np.random.randint(-90, 90)
        # 小黄人粘贴到背景图上的坐标点
        x1_random = np.random.randint(0, 224 - np.maximum(width_random, height_random))
        y1_random = np.random.randint(0, 224 - np.maximum(width_random, height_random))

        bg_photo = pimg.open(os.path.join(bg_path, photo))
        minions_photo = pimg.open("{0}/{1}.png".format(minions_path, minios_random))
        minions_resize = minions_photo.resize((width_random, height_random))
        minions_rotate = minions_resize.rotate(rotate_random)
        r, g, b, alpha = minions_rotate.split()
        bg_photo.paste(minions_rotate, (x1_random, y1_random), mask=alpha)
        bg_photo.save("{0}/{1}.png".format(save_path,
                                           str(index) + "." + str(x1_random) + "." + str(y1_random) + "." + str(
                                               x1_random + width_random) + "." + str(y1_random + height_random) + ".1"))
        index += 1


if __name__ == '__main__':
    bg_imgs_train = r"./datasets/bg_imgs_train"
    bg_imgs_test = r"./datasets/bg_imgs_test"
    minions = r"./datasets/minions"
    train_img = r"./datasets/train_img"
    test_img = r"./datasets/test_img"

    # 创建训练数据
    createBgPhoto(bg_imgs_train, train_img)
    pasteMinions(train_img, minions, train_img)
    # 创建测试数据
    # createBgPhoto(bg_imgs_test, test_img)
    # pasteMinions(test_img, minions, test_img)
