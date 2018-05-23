from flask import Flask, jsonify, request
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import torch
import base64
from flask_cors import CORS
from io import BytesIO
import numpy as np
import re
from model.espcn import ESPCN
import os

app = Flask(__name__)
CORS(app)

def base642im(img_string):
    """Decodes Base64 string to an image array"""
    img_string = re.sub('^data:image/.+;base64,', '', img_string)
    im = Image.open(BytesIO(base64.b64decode(img_string)))
    return im

def im2base64(im):
    """Ecodes image array to Base64"""
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_str = "data:image/png;base64," + str(base64.b64encode(buffered.getvalue()), 'utf-8')
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

def detect(im):
    im.save('tmp/tmp.bmp')
    os.system("cd model;python detect.py --det det")
    im = Image.open('model/det/det_tmp.bmp')
    return im    


@app.route('/sr',methods=['POST'])
def sr_router():
    data = request.json['data']
    scale = request.json['scale']
    im = base642im(data)
    im = sr(im, scale)
    data_scaled = im2base64(im)
    return jsonify({'result': data_scaled})

@app.route('/detect',methods=['POST'])
def detect_router():
    data = request.json['data']
    im = base642im(data)
    im = detect(im)
    data_scaled = im2base64(im)
    return jsonify({'result': data_scaled})


if __name__ == '__main__':
    global espcn 
    espcn = ESPCN()
    espcn.load_state_dict(torch.load('model/espcn.pth'))
    espcn.cuda()
    app.run(port=8000)
