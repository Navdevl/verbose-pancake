import glob
import os
import string
from core.captcha_sequence import CaptchaSequence


if __name__ == '__main__':
    filelist = glob.glob(os.path.join('training_data', "*.png"))
    for f in filelist:
        os.remove(f)
    print("Cleaning successfully completed.")

    characters = string.digits
    cs = CaptchaSequence(characters, batch_size=10, steps=1, save_image=True)
    for X, y in cs:
        pass
