import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional


class Resize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        """
        resize and pad main function
        :param img: np.ndarray
        :return:
        """
        if not isinstance(img, np.ndarray):
            raise ValueError('Only numpy array image supported to resize')
        size = img.shape
        if len(size) < 2 or len(size) > 3:
            raise ValueError('incorrect image data')
        img = cv2.resize(img, dsize=self.size, interpolation=self.interpolation)
        return img


class ResizePadOpenCV(object):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        """

        :param size: (w, h)
        :param interpolation: reference in opencv
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        """
        resize and pad main function
        :param img: np.ndarray
        :return:
        """
        if not isinstance(img, np.ndarray):
            raise ValueError('Only numpy array image supported to resize')
        new_w, new_h = self.size
        size = img.shape
        if len(size) == 4:
            raise ValueError('batch image is not support')
        elif len(size) == 3:
            h, w, c = size
        elif len(size) == 2:
            h, w = size
        else:
            raise ValueError('incorrect image data')

        pad = [0, 0, 0, 0]
        if h / w < new_h / new_w:
            h = int(h * (new_w / w))
            w = int(new_w)
            pad[1] = int((new_h - h) / 2)  # pad top
            pad[3] = new_h - pad[1] - h  # pad bottom
        else:
            w = int(w * (new_h / h))
            h = int(new_h)
            pad[0] = int((new_w - w) / 2)  # pad left
            pad[2] = new_w - pad[0] - w  # pad right

        img = cv2.resize(img, dsize=(w, h), interpolation=self.interpolation)
        img = np.pad(img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)))
        return img


class ResizePadPil(torch.nn.Module):

    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        """
        resize and pad main function
        :param img: pillow Image or torch Tensor
        :return:
        """
        h, w = 0, 0
        new_w, new_h = self.size
        if isinstance(img, Image.Image):
            w, h = img.size
        elif isinstance(img, torch.Tensor):
            size = img.size()
            if len(size) == 4:
                raise ValueError('batch image is not support')
            elif len(size) == 3:
                c, h, w = size
            elif len(size) == 2:
                h, w = size
            else:
                raise ValueError('incorrect image data')
        pad = [0, 0, 0, 0]
        if h / w < new_h / new_w:
            h = int(h * (new_w / w))
            w = int(new_w)
            pad[1] = int((new_h - h) / 2)  # pad top
            pad[3] = new_h - pad[1] - h    # pad bottom
        else:
            w = int(w * (new_h / h))
            h = int(new_h)
            pad[0] = int((new_w - w) / 2)  # pad top
            pad[2] = new_w - pad[0] - w    # pad bottom
        img = functional.resize(img, [h, w], self.interpolation)
        img = functional.pad(img, tuple(pad), padding_mode='constant')
        return img


if __name__ == "__main__":
    test_img = cv2.imread("../test.jpg")
    resize_pad = ResizePadOpenCV(size=(320, 320))
    test_img = resize_pad(test_img)
    print(test_img.shape)
    cv2.imshow("test", test_img)
    cv2.waitKey(0)
