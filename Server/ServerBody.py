import io
import torch

from flask import Flask
from flask import request, send_file, render_template
from numpy import byte
from torchvision.io import read_image
from werkzeug.utils import secure_filename

from Inference import InferenceImage, Inference, tensor2image
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
state_dict = torch.load("backbones/pretrained/matting_weight.pt")
network.load_state_dict(state_dict)
upload_path = './fileUpload/'

@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/composite')
def predict():
    return render_template('composite.html')

@app.route('/filePredict', methods=['GET','POST'])
def predict_file():
    if request.method =='POST':
        file = request.files['img']
        img_bytes = file.read()
        alpha_map = InferenceImage(img_bytes, network)
        alpha_map = tensor2image(alpha_map)
        alpha_map = generate_png(alpha_map)
        file_name="data.png"

        a = BytesIO(alpha_map)
        img = Image.open(a)
        img.save(upload_path+secure_filename(file_name))

    return send_file(BytesIO(alpha_map),
                     as_attachment=True, download_name=file_name)

@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method =='POST':

        f = request.files['img']
        f.save(upload_path+secure_filename(f.filename))
        return 'upload 디렉토리 -> 파일 업로드 성공!'

def generate_png(pil_image):
    byteIO = BytesIO()
    pil_image.save(byteIO, "PNG")
    byteIO.seek(0)
    pil_image = byteIO.read()
    return pil_image

if __name__=='__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', debug=True)

