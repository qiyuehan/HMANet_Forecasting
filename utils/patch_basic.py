import torch
# from my_models.forecasting.linear_fore import Linear_forecasting
from utils import backbone


class Patch_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'pretrain': backbone,
            # "linear_fore": Linear_forecasting,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Use Cuda!')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
