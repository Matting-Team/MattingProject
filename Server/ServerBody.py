import torch

from flask import Flask
from flask import request, send_file, render_template
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
def home():
    return render_template('home.html')

@app.route('/upload')
def render_file():
    return render_template('upload.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/filePredict', methods=['GET','POST'])
def predict_file():
    if request.method =='POST':
        file = request.files['file']
        img_bytes = file.read()
        alpha_map = InferenceImage(img_bytes, network)
        alpha_map = tensor2image(alpha_map)
        alpha_map = generate_png(alpha_map)
        file_name="date.png"

    return send_file(BytesIO(alpha_map),
                     as_attachment=True, attachment_filename=file_name)

@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method =='POST':
        f = request.files['file']
        f.save(upload_path+secure_filename(f.filename))
        return 'upload Directory -> File Upload Done!'

@app.route('/crop', methods = ['GET', 'POST'])
def crop_file():
    if request.method =='POST':
        value1 = int(request.form['value1'])
        value2 = int(request.form['value2'])
        value3 = int(request.form['value3'])
        value4 = int(request.form['value4'])

        f = request.files['file']
        image = Image.open(f)
        cropImage = image.crop((value1, value2, value3, value4))
        cropImage.save(upload_path+secure_filename('crop.png'))

        # return 'upload 디렉토리 -> crop 이미지 파일 업로드 성공!'
        return send_file('fileUpload/crop.png', mimetype='image/jpg')

@app.route('/contact')
def contact():
    return render_template('info.html')

def generate_png(pil_image):
    byteIO = BytesIO()
    pil_image.save(byteIO, "PNG")
    byteIO.seek(0)
    pil_image = byteIO.read()
    return pil_image
if __name__=='__main__':
    app.run(debug=True)

