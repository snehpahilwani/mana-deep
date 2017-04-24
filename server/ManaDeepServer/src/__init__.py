#!flask/bin/python
from flask import Flask
from flask import request

import os
from dominate.tags import base
from flask import send_file

from PIL import Image
import cStringIO
import numpy as np


app = Flask(__name__)
mod = None

@app.route('/')
def index():
    global mod
    if mod == None:
        import model as m
        mod = m
    return "Hello, World!"


@app.route('/predict', methods=['POST','GET'])
def predict():
    global mod
    if mod == None:
        import model as m
        mod = m
    img_png_b64= request.form['img']
    return mod.predict(img_png_b64.split(',')[1])


@app.route('/img', methods=['POST','GET'])
def getimg():
    img_name= '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train/' +  request.args.get('img')
    pil_img = Image.open(img_name)
    pil_img = Image.fromarray((np.array(pil_img)).astype('uint32'))
    #pil_img.mode = 'I'
    #pil_img.point(lambda i:i*(1./256)).convert('L')
    img_io = cStringIO.StringIO()
    pil_img.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response
            
if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0')