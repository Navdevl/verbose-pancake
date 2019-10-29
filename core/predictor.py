import numpy as np
import glob
import os
from PIL import Image
import string
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPooling2D, Flatten, Input
from tensorflow.keras.models import Model


class Predictor(object):
    def __init__(self, height, width):
        self.characters = string.digits
        self.n_class = len(self.characters)
        self.height = height
        self.width = width
        self.n_len = 6
        self.input_tensor = Input((height, width, 3))
        self.create_model()
        self.data = self.load_from_files()

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

    def decode(self, y):
        y = np.argmax(np.array(y), axis=2)[:, 0]
        return ''.join([self.characters[x] for x in y])

    def predict(self):
        self.model.load_weights('model/cnn.h5')
        success = 0
        for X, y in self.data:
            print(X)
            print(y)
            input()
            y_pred = self.model.predict(X)
            print('real: %s\npred: %s' % (self.decode(y), self.decode(y_pred)))
            if (self.decode(y) == self.decode(y_pred)):
                success += 1
        print(success)

    def load_from_files(self, dir='test_data'):
        print("Load from files..")
        test_data = []
        filelist = glob.glob(os.path.join(dir, "*.png"))
        print(filelist)
        for file in filelist:
            test_data.append(self.load_from_file(file))
        return test_data

    def load_from_file(self, filename):
        im = Image.open(filename).convert('RGB')
        im = im.resize((self.width, self.height))
        im_np = np.array(im) / 255.0
        im_np = im_np.reshape((1, self.height, self.width, 3))
        print(filename.split('/')[-1].split('.')[0])
        return (im_np, filename.split('/')[-1].split('.')[0])
