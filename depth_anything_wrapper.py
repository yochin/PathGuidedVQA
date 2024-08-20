import torch
# from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


class depth_anything():
    def __init__(self, encoder='vitl', dataset='vkitti', max_depth=80):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        # encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
        # dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
        # max_depth = 80 # 20 for indoor model, 80 for outdoor model

        self.model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        self.model.load_state_dict(torch.load(f'Depth_Anything_V2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
        self.model.to(DEVICE).eval()

        
    def infer_image(self, image):
        depth_HW = self.model.infer_image(image)

        return depth_HW