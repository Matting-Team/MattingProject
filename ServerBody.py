import io
import cv2
import os
import numpy as np
import torch

import flask
from flask import Flask, redirect, url_for
from flask import request, send_file, render_template
from numpy import byte
from torchvision.io import read_image
from werkzeug.utils import secure_filename

from Inference import InferenceImage, tensor2image, CompositeImage
from models.network import CamNet
from frameworks.ImageLoader import load_singular_image
from frameworks.Utils import print_tensor

from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from io import BytesIO
from PIL import Image


app = Flask(__name__)
network = CamNet(input_c=3, base_c=32, pretrained=True)
network.eval()

user_dict = {}

state_dict = torch.load("backbones/pretrained/matting_weight.pt")
network.load_state_dict(state_dict)
upload_path = './static/fileUpload/'

#options...
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def intro():
    return render_template('intro.html')

# render predict page
@app.route('/home')
def home():
    return render_template('home.html')

# Render composite page
@app.route('/composite')
def composite():
    return render_template('composite.html')

#render download page
@app.route('/download', methods=['GET', 'POST'])
def down():
    ip_addr = flask.request.remote_addr
    path = os.path.join(upload_path, "{}data.png".format(ip_addr))

    return render_template('download_page.html', user_image=path)

#download(comp) page
@app.route('/download_comp', methods=['GET', 'POST'])
def down_comp():
    ip_addr = flask.request.remote_addr
    path = os.path.join(upload_path, "{}result.png".format(ip_addr))

    return render_template('comp_download_page.html', user_image=path)

# display img...
@app.route('/display')
def display_image():
    ip_addr = flask.request.remote_addr
    filename = "{}data.png".format(ip_addr)
    return redirect(url_for('static', filename='fileUpload/' + filename), code=301)

# display composite img...
@app.route('/display_comp')
def display_comp_image():
    ip_addr = flask.request.remote_addr
    filename = "{}result.png".format(ip_addr)
    return redirect(url_for('static', filename='fileUpload/' + filename), code=302)

# by POST message, predict img
@app.route('/filePredict', methods=['GET','POST'])
def predict_file():
    if request.method =='POST':
        ip_addr = flask.request.remote_addr

        file = request.files['img']
        img_bytes = file.read()
        alpha_map = InferenceImage(img_bytes, network, img_t=True)
        alpha_map = tensor2image(alpha_map)
        alpha_map = generate_png(alpha_map)

        file_name="{}data.png".format(ip_addr)
        a = BytesIO(alpha_map)
        img = Image.open(a)
        img.save(upload_path+secure_filename(file_name))


    return redirect("/download")

# Composite & save result
@app.route('/fileComposite', methods = ['GET', 'POST'])
def composite_file():
    if request.method =='POST':
        ip_addr = flask.request.remote_addr

        f = request.files['img']
        filename = "{}back.png".format(ip_addr)
        f.save(upload_path+secure_filename(filename))

        data = upload_path+"/{}data.png".format(ip_addr)
        back = upload_path+"/{}back.png".format(ip_addr)
        d = Image.open(data)
        b = Image.open(back)

        f_data = d.resize((int(b.width), int(b.height)))
        f_data.save(upload_path+secure_filename("{}fore.png".format(ip_addr)))
        fore = upload_path+"/{}fore.png".format(ip_addr)
        with open(fore, 'rb') as f:
            fore_img = f.read()

        with open(back, 'rb') as f:
            back_img = f.read()

        com_img = CompositeImage(fore_img, back_img, network)
        com_img = tensor2image(com_img)
        com_img = generate_png(com_img)

        result = BytesIO(com_img)
        result = Image.open(result)

        result.save(upload_path+secure_filename("{}result.png".format(ip_addr)))

    return redirect("/download_comp")

# download matting image
@app.route('/download_a', methods = ['GET'])
def download_fg():
    ip_addr = flask.request.remote_addr
    path = upload_path + "/{}data.png".format(ip_addr)
    if os.path.isfile(path):
        return send_file(path, as_attachment=True, attachment_filename="result.png", cache_timeout=0)
    else:
        return None

# download composite image
@app.route('/download_c', methods = ['GET'])
def download_comp():
    ip_addr = flask.request.remote_addr
    path = upload_path + "/{}result.png".format(ip_addr)
    if os.path.isfile(path):
        return send_file(path, as_attachment=True, attachment_filename="comp_result.png", cache_timeout=0)
    else:
        return None

# after predict, image is RGB format.
# Change to RGBA format is required
def gen_real_alpha(img):
    rgba = img.convert("RGBA")
    datas = rgba.getdata()
    newData = []
    for item in datas:
        if item[0] == 0.0 and item[1] == 0.0 and item[2] == 0.0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    rgba.putdata(newData)
    return rgba

# remove images...(clean storage)
def remove_imgs(ip_addr):
    file_list = os.listdir(upload_path)

    for file_name in file_list:
        if ip_addr in file_name:
            path = os.path.join(upload_path, file_name)
            if os.path.isfile(path):
                os.remove(path)

# pil image to png image
def generate_png(pil_image):
    pil_image = gen_real_alpha(pil_image)
    byteIO = BytesIO()
    pil_image.save(byteIO, "PNG")
    byteIO.seek(0)
    pil_image = byteIO.read()

    return pil_image



if __name__=='__main__':
    app.run(debug=True)
