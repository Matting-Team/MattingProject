import Models.SegMatting.frameworks.loader as dataloader
import Utils.reader as yamlreader
from Utils.BasicUtil import *

config = yamlreader.Opener("frameworks/config.yaml")
training_config = config["Training"]
loader_config = config["LOADER"]
network_config = config["NETWORK"]

loader = dataloader.TorchLodader(loader_config)

