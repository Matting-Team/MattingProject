import torch
import numpy as np
from models.network import CamNet
from frameworks.ImageLoader import load_singular_image
from frameworks.Utils import print_tensor
import torchvision.transforms as transforms
import PIL.Image as Image
import io


def transform_byte_image(image_byte):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(io.BytesIO(image_byte))
    image = image.resize([image.size[0]//32*32, image.size[1]//32*32])
    image = np.array(image)[..., :3]
    return transform(image).unsqueeze(0)

# change tensor to image
##############################################################
def tensor2image(tensor, alpha=None):
    tensor = tensor.squeeze(0)

    if alpha is not None:
        alpha = alpha.squeeze(0)
        torch.cat([tensor, alpha], dim=0)
    if tensor.shape[0] == 3:
        image = transforms.ToPILImage(mode="RGB")(tensor)
    elif tensor.shape[0]==4:
        image = transforms.ToPILImage(mode="RGBA")(tensor)
    else:
        image = transforms.ToPILImage(mode="L")(tensor)
    return image


#
# InferenceImage --> It takes image bytes as input and extracts the alpha map.
# input -> Image byte obtained through flask, pretrained matting network
#######################################################################
def InferenceImage(image_byte, network, img_t=False):
    tensor = transform_byte_image(image_byte)
    result = network(tensor)
    if isinstance(result, list):
        alpha = result[-1]
        image = alpha*(tensor*0.5+0.5)
    else:
        alpha = result
        image = alpha*(tensor*0.5 + 0.5)
        if img_t:
            image = torch.cat([image, alpha], dim=1)
    return image

# Composite Foreground Image and Background
# input -> Foreground, background bytes obtained through flask. pretrained mating network
###############################################################
def CompositeImage(image_byte, background_byte, network):
    tensor = transform_byte_image(image_byte)
    background = transform_byte_image(background_byte)
    result = network(tensor)
    if isinstance(result, list):
        alpha = result[-1]
        image = composite(tensor, alpha, background)
    else:
        alpha = result
        image = composite(tensor, alpha, background)

    return image

# tensor, alpha map을 입력받아 합성을 수행하는 코드.
############################################################
def composite(image, alpha, background):
    image = image*0.5+0.5
    background = background*0.5+0.5
    comp_image = image*alpha + background*(1-alpha)
    return comp_image