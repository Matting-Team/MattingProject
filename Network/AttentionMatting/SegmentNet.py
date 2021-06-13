import torch
import torch.nn as nn
import torchvision
import Network as net
#####################################33
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
######################################

def print_image(img):
    img = img.detach().cpu()
    img = torchvision.utils.make_grid(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.transpose(img, (1, 2, 0))
    img = img*std + mean
    plt.imshow(img)
    plt.show()

model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
model = model.features.eval()#나중에 to(device 추가)
test_img = torch.randn((1,3, 128, 128))
x = test_img

#####################################3
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
'''
######################################
dog = Image.open('dog.jpg')
input = transform(dog)
input = input.unsqueeze(0)
print(input.shape)
print_image(input)
'''
class ConvTBatch(nn.Module):
    def __init__(self, c, kernel_size = 4, stride=2, padding=1):
        super(ConvTBatch, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c//2)
        )
    def forward(self, input):
        return self.seq(input)
class Encoder(nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model

    def forward(self, input):
        x=input
        res_list = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, torchvision.models.densenet._Transition):
                res_list.append(x)
        return x, res_list

'''
encoder = Encoder(model)
result, l = encoder(input)
for i in l:
    print(i.shape)
print(result.shape)
'''
class Decoder(nn.Module):
    def __init__(self, depth=5):
        super(Decoder, self).__init__()
        self.m_list = nn.ModuleList([])
        self.firstConv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        c= 512
        for idx in range(depth):
            m = nn.Sequential(
                ConvTBatch(c),
                nn.ReLU(inplace=True),
                net.ResBlock(c//2),
                nn.ReLU(inplace=True)
            )
            self.m_list.append(m)
            c=c//2
        self.lastConv = nn.Sequential(
            nn.Conv2d(16, 1, 7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, input, skip_con=[]):
        x = self.firstConv(input)
        idx = 0
        for m in self.m_list:
            idx += 1
            if idx < 4:
                x = x + skip_con[3 - idx]
            x = m(x)
        x = self.lastConv(x)
        return x

class DoubleDecoder(nn.Module):
    def __init__(self):
        super(DoubleDecoder, self).__init__()
        self.fore_decoder = Decoder()
        self.back_decoder = Decoder()
    def forward(self, input, skip_connection):
        fg = self.fore_decoder(input, skip_connection)
        bg = self.back_decoder(input, skip_connection)
        return fg, bg
'''
decoder = Decoder()
result = decoder(result, l)
print(result.shape)
'''