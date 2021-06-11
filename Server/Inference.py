import torch
import numpy as np
from Server.models.network import CamNet
from Server.frameworks.ImageLoader import load_singular_image
from Server.frameworks.Utils import print_tensor
import torchvision.transforms as transforms
import PIL.Image as Image
import io
def Inference():
    # Load Configs --> base c, input c, path etc...
    path = "D:/Dataset/MattingTest/own_image"
    name = "woman.jpg"
    network = CamNet(input_c=3, base_c=32, pretrained=True)
    state_dict = torch.load("backbones/pretrained/matting_weight.pt")
    network.load_state_dict(state_dict)
    input = load_singular_image(path, name)

    segment, detail, alpha = network(input)
    print_tensor(alpha)

def transform_byte_image(image_byte):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(io.BytesIO(image_byte))

    image = image.resize([image.size[0]//32*32, image.size[1]//32*32])
    image = np.array(image)[..., :3]
    return transform(image).unsqueeze(0)

def tensor2image(tensor):
    tensor = tensor.squeeze(0)
    if tensor.shape[0] == 3:
        #tensor = ((tensor*0.5)+0.5)
        image = transforms.ToPILImage(mode="RGB")(tensor)
    else:
        #tensor = ((tensor * 0.5) + 0.5)
        image = transforms.ToPILImage(mode="L")(tensor)
    return image



def InferenceImage(image_byte, network):
    tensor = transform_byte_image(image_byte)
    result = network(tensor)
    if isinstance(result, list):
        alpha = result[-1]
    else:
        alpha = result

    print(alpha.min())
    print(alpha.max())
    return alpha
