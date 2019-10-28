import requests
import shutil


class VDSCreator(object):
    def __init__(self):
        self.url = "https://ipindiaonline.gov.in/eregister/captcha.ashx"

    def download(self, filename):
        response = requests.get(self.url, stream=True)
        with open('test_data/{0}.png'.format(filename), 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
