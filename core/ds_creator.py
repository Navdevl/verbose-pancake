from PIL import Image, ImageFont, ImageDraw, ImageOps
import random


class DSCreator(object):
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.font_location = 'font/Helvetica-Normal.ttf'
        self.fontsize = 30

    def init_new_image(self):
        self.image = Image.new("RGB", (135, 35), (255, 255, 255))

    def create_noise_curve(self, color):
        w, h = self.image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 10), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h)
        points = [x1, y1, x2, y2]
        end = random.randint(90, 200)
        start = random.randint(45, 90)
        ImageDraw.Draw(self.image).arc(points, start, end, fill=color, width=2)

    def draw_character(self, c, font):
        color = (255, 255, 255,)
        draw = ImageDraw.Draw(self.image)
        w, h = draw.textsize(c, font=font)
        dx = random.randint(0, 2)
        dy = random.randint(0, 6)
        im = Image.new('RGB', (w + dx, h + dy))
        ImageDraw.Draw(im).text((dx, dy), c, font=font, fill=color)

        # rotate
        # im = im.crop(im.getbbox())
        im = im.rotate(random.uniform(-30, 45), Image.BICUBIC, expand=1)

        # warp
        dx = w * random.uniform(0.1, 0.2)
        dy = h * random.uniform(0.2, 0.3)
        x1 = int(random.uniform(-dx, dx))
        y1 = int(random.uniform(-dy, dy))
        x2 = int(random.uniform(-dx, dx))
        y2 = int(random.uniform(-dy, dy))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        # data = (
        #     x1, y1,
        #     -x1, h2 - y2,
        #     w2 + x2, h2 + y2,
        #     w2 - x2, -y1,
        # )
        # im = im.transform((w, h), Image.QUAD, data)
        im = im.resize((w2, h2), Image.HAMMING)
        return im

    def create_image(self, random_str):
        self.init_new_image()
        font = ImageFont.truetype(self.font_location, self.fontsize)

        images = []
        for c in random_str:
            images.append(self.draw_character(c, font))

        text_width = sum([im.size[0] for im in images])
        average = int(text_width / len(random_str))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for index, im in enumerate(images):
            w, h = im.size
            inverted_image = ImageOps.invert(im)
            self.image.paste(inverted_image, (offset, int((self.height - h) / 2)))
            offset = offset + w + random.randint(-rand, 0)

        self.create_noise_curve((0, 0, 0,))
        return self.image.resize((135, 35), Image.ANTIALIAS)
