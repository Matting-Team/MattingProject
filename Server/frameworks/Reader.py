import yaml

# Read Config File (Yaml)
################################################################
class Opener:
    def __init__(self, path):
        self.path = path
        self.conf=None
        self.loader_config = None
        self.network_config = None
        with open(path) as f:
            self.conf = yaml.safe_load(f)
            self.loader_config = self.conf['Loader']
            self.network_config = self.conf['Network']

    def info(self):
        if self.conf is not None:
            print('length:---%d---'%(len(self.conf)))
            print('keys:---%s---'%(self.conf.keys().__str__()))

# Not Used...
class ConfigManager:
    def __init__(self, conf):
        self.length = len(conf)
        self.keys = conf.keys()
