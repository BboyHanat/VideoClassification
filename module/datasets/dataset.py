import os
import cv2
import json
import torch
import numpy as np
from pathlib import PurePosixPath
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from module.datasets import BaseDataset
from utils.resize_utils import ResizePadOpenCV, Resize


class VideoDataset(Dataset):

    def __init__(self,
                 data_root,
                 clip_len=8,
                 img_h=320,
                 img_w=320,
                 frame_sample_rate=1,
                 resize_keep_ratio=True,
                 mode='train'):

        folder = PurePosixPath(data_root, mode)  # get the directory of the specified split
        self.clip_len = clip_len
        self.img_h = img_h
        self.img_w = img_w
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
        self.resize_keep_ratio = resize_keep_ratio
        self.file_names, labels = list(), list()
        if resize_keep_ratio:
            self.resize = ResizePadOpenCV(size=(img_w, img_h))
        else:
            self.resize = Resize(size=(img_w, img_h))

        files = [f for f in os.listdir(folder) if '.DS' not in f]
        for label in sorted(files):
            for file in os.listdir(os.path.join(folder, label)):
                self.file_names.append(os.path.join(folder, label, file))
                labels.append(label)

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        self.mean = np.reshape(np.array([0.485, 0.456, 0.406], dtype=np.float32), newshape=(1, 1, 1, 3))
        self.std = np.reshape(np.array([0.229, 0.224, 0.225], dtype=np.float32), newshape=(1, 1, 1, 3))

        label_file = 'class_labels.json'
        with open(label_file, 'w') as f:
            json.dump(self.label2index, f)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        frames = self.load_video(self.file_names[index])
        while frames.shape[0] < self.clip_len + 2:
            index = np.random.randint(self.__len__())
            frames = self.load_video(self.file_names[index])

        frames = self.crop(frames, self.clip_len)
        frames = self.normalize(frames)
        frames = self.to_tensor(frames)
        frames = torch.tensor(frames, dtype=torch.float32)
        labels = torch.tensor(self.label_array[index])

        return frames, labels

    def load_video(self, file_name):
        """
        load video and return sampled video clip
        :param file_name: video path
        :return: sampled video clip ndarray
        """
        remainder = np.random.randint(self.frame_sample_rate)
        capture = cv2.VideoCapture(file_name)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        buffer = np.zeros((frame_count_sample, self.img_h, self.img_w, 3), np.dtype('float32'))

        count = 0
        start_idx = np.random.randint(frame_count - self.frame_sample_rate * self.clip_len) if \
            frame_count > self.frame_sample_rate * self.clip_len else 0
        sample_count = 0
        # read in each frame, one at a time into the numpy buffer array
        while count < frame_count:
            ret, frame = capture.read()
            if not ret or sample_count >= frame_count_sample:
                break
            if count < start_idx:
                count += 1
                continue
            if count % self.frame_sample_rate == remainder:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.resize(frame)
                buffer[sample_count, :, :, :] = frame
                sample_count += 1
            count += 1
        capture.release()
        return buffer

    def normalize(self, frames):
        frames = frames / 255.0
        frames = (frames - self.mean) / self.std
        return frames

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def to_tensor(buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    @staticmethod
    def crop(buffer, clip_len):
        time_index = np.random.randint(buffer.shape[0] - clip_len + 1)
        buffer = buffer[time_index:time_index + clip_len, :, :, :]
        return buffer


def dataloader_test():

    data_root = '/Users/BboyHanat/workspace/datasets/UCF-101'
    dataset = VideoDataset(data_root, mode='train')
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
    for step, (buffer, label) in enumerate(train_dataloader):
        print("label: {}".format(label.shape), "buffer: {}".format(buffer.shape))


if __name__ == '__main__':
    dataloader_test()

