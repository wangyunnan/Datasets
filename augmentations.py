import random
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image
import PIL.ImageFilter as ImageFilter
import PIL.ImageEnhance as ImageEnhance

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, lbl):
        assert img.size == lbl.size

        ori = img.copy()
        for augmentation in self.augmentations:
            img, lbl, ori= augmentation(img, lbl, ori)

        return img, lbl, ori

class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation):
        self.brightness = [max(1-brightness, 0), 1+brightness]
        self.contrast = [max(1-contrast, 0), 1+contrast]
        self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, img, lbl, ori):
        assert img.size == lbl.size
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        return img, lbl, ori

class RandomMirror(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, lbl, ori):
        assert img.size == lbl.size
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
            ori = ori.transpose(Image.FLIP_LEFT_RIGHT)

        return img, lbl, ori

class RandomScale(object):
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, img, lbl, ori):
        assert img.size == lbl.size
        w, h = img.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        sw, sh = int(w * scale), int(h * scale)
        img = img.resize((sw, sh), Image.BILINEAR)
        lbl = lbl.resize((sw, sh), Image.NEAREST)
        ori = ori.resize((sw, sh), Image.BILINEAR)
        return img, lbl, ori

class RandomGauss(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, lbl, ori):
        assert img.size == lbl.size
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, lbl, ori

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl, ori):
        assert img.size == lbl.size
        w, h = img.size
        cw, ch = self.size

        if (cw, ch) == (w, h): return img, lbl, ori
        x, y = random.random() * (w - cw), random.random() * (h - ch)
        margin = int(x), int(y), int(x) + cw, int(y) + ch
        img = img.crop(margin)
        lbl = lbl.crop(margin)
        ori = ori.crop(margin)

        return img, lbl, ori

class RandomAffine(object):
    def __init__(self, p=0.5, operations=('sx', 'sy', 'r', 'tx', 'ty')):
        self.p = p
        self.operations = operations

    def __call__(self, img, lbl, logits):
        device = img.device
        b, c, h, w = img.shape
        lbl = lbl.float().unsqueeze(1)

        all_theta = []
        for i in range(b):
            theta = torch.eye(n=2, m=3).float().unsqueeze(0)
            if random.random() < self.p:
                theta = self.random_operation((h, w))
            all_theta.append(theta)
        all_theta = torch.cat(all_theta)

        loss_mask = torch.ones((b, 1, h, w)).float().to(device)
        grid = F.affine_grid(all_theta, img.size(), align_corners=True).to(device)

        new_img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
        new_lbl = F.grid_sample(lbl, grid, mode='nearest', align_corners=True).long().squeeze(1)
        new_logits = F.grid_sample(logits, grid, mode='bilinear', align_corners=True)
        new_loss_mask = F.grid_sample(loss_mask, grid, mode='nearest', align_corners=True).long().squeeze(1)

        new_lbl[(1 - new_loss_mask).bool()] = 255

        return new_img, new_lbl, new_logits, new_loss_mask

    def random_operation(self, size):
        h, w = size
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float)
        theta = self.getAffineMatrix(M, (h, w))
        opration = random.choice(self.operations)
        # ShearX
        if opration == 'sx':
            angle = random.uniform(-15, 15) / 180 * np.pi
            M = np.array([[1, np.tan(angle), 0], [0, 1, 0]], dtype=np.float)
            theta = self.getAffineMatrix(M, (h, w))
        # ShearY
        elif opration == 'sy':
            angle = random.uniform(-15, 15) / 180 * np.pi
            M = np.array([[1, 0, 0], [np.tan(angle), 1, 0]], dtype=np.float)
            theta = self.getAffineMatrix(M, (h, w))
        # Rotate
        elif opration == 'r':
            angleR = random.uniform(-15, 15) / 180 * np.pi
            M = np.array([
                    [np.cos(angleR), -np.sin(angleR), 0],
                    [np.sin(angleR), np.cos(angleR), 0],
                    [0, 0, 1]], dtype=np.float)
            theta = torch.from_numpy(M)
        # TranslateX
        elif opration == 'tx':
            translateX = random.uniform(-0.3, 0.3)
            M = np.array([[1, 0, translateX], [0, 1, 0], [0, 0, 1]]).astype(np.float)
            theta = torch.from_numpy(M)
        # TranslateY
        elif opration == 'ty':
            translateY = random.uniform(-0.3, 0.3)
            M = np.array([[1, 0, 0], [0, 1, translateY], [0, 0, 1]]).astype(np.float)
            theta = torch.from_numpy(M)

        return theta[:2, :].float().unsqueeze(0)

    def getAffineMatrix(self, M, size):
        H, W = size
        T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
        M = np.vstack((M, np.asarray([[0, 0, 1]])))
        theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
        theta = torch.from_numpy(theta)
        return theta

class RandomGray(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, lbl, ori):
        assert img.size == lbl.size
        if random.random() < self.p:
            img = img.convert('L').convert('RGB')
        return img, lbl, ori

class RandomCutMix(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, source_img, source_lbl, source_logits, target_img, target_lbl, target_logits):
        assert source_img.shape == target_img.shape
        assert source_lbl.shape == target_lbl.shape
        assert source_logits.shape == target_logits.shape
        device = source_img.device
        b, c, h, w = source_img.shape
        source_lbl = source_lbl.float().unsqueeze(1)
        target_lbl = target_lbl.float().unsqueeze(1)

        all_mask = []
        for i in range(b):
            mix_mask = torch.zeros((h, w)).long().to(device).unsqueeze(0)
            if random.random() < self.p:
                mix_mask = self.generate_mask((h, w)).to(device).unsqueeze(0)
            all_mask.append(mix_mask)
        all_mask = torch.cat(all_mask).unsqueeze(1)
        new_source_img = source_img * (1.0 - all_mask) + target_img * all_mask
        new_source_lbl = source_lbl * (1.0 - all_mask) + target_lbl * all_mask
        new_source_logits = source_logits * (1.0 - all_mask) + target_logits * all_mask
        new_target_img = target_img * (1.0 - all_mask) + source_img * all_mask
        new_target_lbl = target_lbl * (1.0 - all_mask) + source_lbl * all_mask
        new_target_logits = target_logits * (1.0 - all_mask) + source_logits * all_mask

        return new_source_img, new_source_lbl.long().squeeze(1), new_source_logits, \
               new_target_img, new_target_lbl.long().squeeze(1), new_target_logits

    def generate_mask(self, img_size, ratio=2):
        cutout_area = img_size[0] * img_size[1] / ratio

        w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
        h = np.round(cutout_area / w)

        x_start = np.random.randint(0, img_size[1] - w + 1)
        y_start = np.random.randint(0, img_size[0] - h + 1)
        x_end = int(x_start + w)
        y_end = int(y_start + h)

        mask = torch.zeros(img_size)
        mask[y_start:y_end, x_start:x_end] = 1
        return mask.long()




def get_augmentations():
    return Compose(
        [
            RandomMirror(p=0.5),
            RandomGray(p=1.)
            #ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            #RandomGauss(p=0.5),
            #RandomScale(scale_range=(0.55, 1.0)),
            #RandomCrop((1024,512)),
        ]
    )

def test_affine():
    augs = RandomAffine()
    soft = cv2.imread('../save/00003_color.png')
    soft = soft.transpose((2, 0, 1))
    img = cv2.imread('../save/00003.png')
    img = img.transpose((2, 0, 1))

    soft = torch.from_numpy(soft).float().unsqueeze(0)
    img = torch.from_numpy(img).float().unsqueeze(0)

    img, soft, mask = augs(img, soft)
    soft = soft.squeeze(0)
    img = img.squeeze(0)
    mask = mask.squeeze(0)

    soft = soft.numpy()
    img = img.numpy()
    mask = mask.numpy()

    soft = np.uint8(soft)
    soft = soft.transpose((1, 2, 0))
    img = np.uint8(img)
    img = img.transpose((1, 2, 0))

    cv2.imwrite('../save/a1.png', soft)
    cv2.imwrite('../save/a2.png', img)
    cv2.imwrite('../save/a3.png', mask)

if __name__ == '__main__':
    import cv2
    import numpy as np

    augs = get_augmentations()
    img = PIL.Image.open('../save/00003.png').convert('RGB')

    img = img.resize((1914, 1052), Image.BILINEAR)
    ori = img.copy()
    lbl = Image.open('../save/00003_color.png')
    lbl = lbl.resize((1914, 1052), Image.NEAREST)
    img, lbl, _ = augs(img, lbl)

    ori.show()


    # img, img, mask = augs(img, soft)
    #
    # img = img.squeeze(0)
    # mask = mask.squeeze(0)
    #
    # img = img.numpy()
    # mask = mask.numpy()
    #
    # soft = np.uint8(soft)
    # soft = soft.transpose((1, 2, 0))
    # img = np.uint8(img)
    # img = img.transpose((1, 2, 0))
    #
    # cv2.imwrite('../save/a1.png', soft)
    # cv2.imwrite('../save/a2.png', img)
    # cv2.imwrite('../save/a3.png', mask)




