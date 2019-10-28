import random
import numpy as np
from tensorflow.keras.utils import Sequence
from captcha.image import ImageCaptcha
from core.ds_creator import DSCreator


class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=6, width=135, height=35, save_image=False):
        print("Initializing the CaptchaSequence.")
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height, font_sizes=[30])
        self.ds_creator = DSCreator(width=width, height=height)
        self.save_image = save_image

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            generated_img = self.ds_creator.create_image(random_str)
            if self.save_image:
                generated_img.save('training_data/{0}.png'.format(random_str))
            X[i] = np.array(generated_img) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y

