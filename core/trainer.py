#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import glob
from PIL import Image
from core.captcha_sequence import CaptchaSequence
from tensorflow.keras.models import Model, Input
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


class Trainer(object):
    def __init__(self, characters, height, width):
        self.characters = characters
        self.n_class = len(characters)
        self.n_len = 6
        self.input_tensor = Input((height, width, 3))
        self.train_data = CaptchaSequence(characters, batch_size=64, steps=1000)
        self.valid_data = self.load_from_files(dir='test_data')
        self.width = width
        self.height = height

    def decode(self, y):
        y = np.argmax(np.array(y), axis=2)[:, 0]
        return ''.join([self.characters[x] for x in y])

    def load_from_files(self, dir):
        test_data = []
        filelist = glob.glob(os.path.join('dir', "*.png"))
        for file in filelist:
            test_data.append(self.load_from_file((file)))

    def load_from_file(self, filename):
        img_path = os.path.join("imgs", filename)
        im = Image.open(img_path).convert('RGB')
        im = im.resize((self.width, self.height))
        im_np = np.array(im) / 1.0
        im_np = im_np.reshape((1, self.height, self.width, 3))
        return (im_np, filename.split('.')[0])

    def create_model(self):
        for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
            for j in range(n_cnn):
                x = Conv2D(32 * 2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(
                    self.input_tensor)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            x = MaxPooling2D(2)(x)

        x = Flatten()(x)
        x = [Dense(self.n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(self.n_len)]
        self.model = Model(inputs=self.input_tensor, outputs=x)

    def train(self, optimizer_value=1e-3):
        callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv'), ModelCheckpoint('cnn_best.h5', save_best_only=True)]
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(optimizer_value, amsgrad=True),
            metrics=['accuracy'])
        self.model.fit_generator(
            self.train_data,
            epochs=10,
            validation_data=self.valid_data,
            workers=8,
            use_multiprocessing=True,
            callbacks=callbacks)

    def execute(self):
        self.train(optimizer_value=1e-3)
        self.train(optimizer_value=1e-4)
        self.save()

    def save(self):
        self.model.save('cnn.h5', include_optimizer=False)

