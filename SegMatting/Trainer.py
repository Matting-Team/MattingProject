import frameworks.loader as dataloader
import frameworks.reader as yamlreader
import frameworks.Utils as utils

config = yamlreader.Opener("frameworks/config.yaml")
training_config = config["Training"]
loader_config = config["LOADER"]
network_config = config["NETWORK"]

loader = dataloader.TorchLodader(loader_config)

