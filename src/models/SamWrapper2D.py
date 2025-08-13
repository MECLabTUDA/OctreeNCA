import torch, einops
import torch.nn as nn
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import numpy as np

class SamWrapper2D():
    def __init__(self, predictor: SamPredictor):
        super().__init__()
        self.predictor = predictor

    def to(self, device):
        return self
    
    def parameters(self):
        return self.predictor.model.parameters()
    
    def train(self):
        return self
    
    def eval(self):
        return self
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor=None, batch_duplication: int=1):
        return self.forward(x, y, batch_duplication)


    def segment_class(self, input_img, seg, seg_class):
        input_prompt_points = []
        while len(input_prompt_points) < 20:
            x = np.random.randint(0, input_img.shape[0])
            y = np.random.randint(0, input_img.shape[1])
            if seg[seg_class, x, y] == 1:
                input_prompt_points.append((y, x))

        input_prompt_points = np.array(input_prompt_points)
        input_prompt_point_labels = np.ones(len(input_prompt_points))


        masks, confidence, _ = self.predictor.predict(input_prompt_points, input_prompt_point_labels)

        return_idx = confidence.argmax()


        #plt.subplot(1, 3, 1)
        #plt.imshow(input_img)
        #plt.gca().scatter(input_prompt_points[:, 0], input_prompt_points[:, 1], color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
        #plt.subplot(1, 3, 2)
        #plt.imshow(seg[seg_class])
        #plt.subplot(1, 3, 3)
        #plt.imshow(masks[return_idx])
        #plt.show()

        return masks[return_idx]

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor=None, batch_duplication: int=1):
        assert y is not None

        in_device = x.device

        x = x.cpu().numpy()
        y = y.cpu().numpy()
        x = einops.rearrange(x, '1 c h w -> h w c')

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        x = x * std + mean
        x *= 255

        x = x.astype(np.uint8)

        y = einops.rearrange(y, '1 c h w -> c h w')

        self.predictor.set_image(x)


        out_segmentation = []
        for c in range(y.shape[0]):
            mask = self.segment_class(x, y, c)
            mask = einops.rearrange(mask, 'h w -> 1 1 h w')
            out_segmentation.append(mask)

        out_segmentation = np.concatenate(out_segmentation, axis=1)
        out_segmentation = einops.rearrange(out_segmentation, '1 c h w -> 1 h w c')

        logits = out_segmentation * 2e4 - 1e4
        

        assert y.shape[0] == 1, "test for other!"

        out_segmentation = torch.tensor(out_segmentation)
        logits = torch.tensor(logits)

        return {"pred": out_segmentation.to(in_device), 
                "logits": logits.to(in_device),
                }