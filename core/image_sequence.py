import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from captcha.image import ImageCaptcha
from core.ds_creator import DSCreator


class ImageSequence(Sequence):
    def __init__(self, dir, characters, width=135, height=35):
        print("Initializing the ImageSequence.")
        self.width = width
        self.dir = dir
        self.n_len = 6
        self.batch_size = 10
        self.n_class = len(characters)
        self.height = height
        self.generator = ImageCaptcha(width=width, height=height, font_sizes=[30])
        self.ds_creator = DSCreator(width=width, height=height)
        self.filelist = glob.glob(os.path.join(dir, "*.png"))
        self.characters = characters

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        data = self.load_from_files(dir=self.dir)
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            X[i] = data[i][0]
            for j, ch in enumerate(data[i][1]):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y

    def load_from_files(self, dir):
        test_data = []
        for file in self.filelist:
            test_data.append(self.load_from_file(file))
        return test_data

    def load_from_file(self, filename):
        im = Image.open(filename).convert('RGB')
        im = im.resize((self.width, self.height))
        im_np = np.array(im) / 255.0
        im_np = im_np.reshape((1, self.height, self.width, 3))
        return (im_np, filename.split('/')[-1].split('.')[0])
