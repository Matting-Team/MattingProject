import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from Utils.reader import Opener
from Models.SegMatting.frameworks.loader import TorchLodader

from Server.models.network import CamNet, Discriminator
from Models.SegMatting.frameworks.TrainModules import evaluation_testset

if __name__=="__main__":
    yaml_reader = Opener("config.yaml")
    pt_path = "C:/Users/김정민/PycharmProjects/SegMatting/backbones/pretrained/matting_weight.pt"
    state_dict = torch.load(pt_path)


    config = yaml_reader.conf
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loader_config = config['Loader']

    loader = TorchLodader(loader_config)
    net = CamNet()
    net.load_state_dict(state_dict['model'])
    net.to(device)

    eval_loader = DataLoader(loader, batch_size=1, shuffle=False)
    with torch.no_grad():
        evaluation_testset(net, eval_loader, device)