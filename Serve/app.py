from flask import Flask, jsonify, request
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import torch
import base64
from io import BytesIO
import numpy as np
from model.espcn import ESPCN
app = Flask(__name__)


def base642im(img_string):
    """Decodes Base64 string to an image array"""
    first_coma = img_string.find(',')
    img_str = img_string[first_coma:].encode('ascii')
    missing_padding = 4 - len(img_str) % 4
    if missing_padding:
        img_str += b'='* missing_padding
    img_bytes = base64.decodestring(img_str)
    im = np.asarray(bytearray(img_bytes), dtype="uint8")
    im = Image.fromarray(im)
    return im

def im2base64(im):
    """Ecodes image array to Base64"""
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def sr(im, scale):
    im = im.convert('YCbCr')
    im, cb, cr = im.split()
    h, w = im.size
    im = ToTensor()(im)
    im = Variable(im).view(1, -1, w, h)
    im = im.cuda()
    with torch.no_grad():
        im = espcn(im)
    im = torch.clamp(im, 0., 1.)
    im = im.cpu()
    im = im.data[0]
    im = ToPILImage()(im)
    cb = cb.resize(im.size, Image.BICUBIC)
    cr = cr.resize(im.size, Image.BICUBIC)
    im = Image.merge('YCbCr', [im, cb, cr])
    im = im.convert('RGB')
    return im


@app.route('/sr',methods=['POST'])
def index():
    data = request.form['data']
    scale = request.form['scale']
    im = base642im(data)
    im = sr(im, scale)
    data_scaled = im2base64(im)
    return jsonify({'result': data_scaled})


if __name__ == '__main__':
    global espcn 
    espcn = ESPCN()
    espcn.load_state_dict(torch.load('model/espcn.pth'))
    app.run(port=8000)