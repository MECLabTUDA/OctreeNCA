import einops
import torch
import numpy as np

class DiceLoss(torch.nn.Module):
    def __init__(self, useSigmoid: bool = True, smooth: float = 1) -> None:
        self.useSigmoid = useSigmoid
        self.smooth = smooth
        super().__init__()

    def compute(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.useSigmoid:
            probabilities = torch.sigmoid(logits)
        probabilities = torch.flatten(probabilities) 
        target = torch.flatten(target)
        
        intersection = (probabilities * target).sum()
        dice_loss = 1 - (2.*intersection + self.smooth)/(probabilities.sum() + target.sum() + self.smooth)
        
        return dice_loss
    

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        assert logits.shape == target.shape
        loss_ret = {}
        loss = 0
        if len(logits.shape) == 5 and target.shape[-1] == 1:
            for m in range(target.shape[-1]):
                loss_loc = self.compute(logits[..., m], target[...])
                loss = loss + loss_loc
                loss_ret[f"mask_{m}"] = loss_loc.item()
        else:
            for m in range(target.shape[-1]):
                if 1 in target[..., m]:
                    loss_loc = self.compute(logits[..., m], target[..., m])
                    loss = loss + loss_loc
                    loss_ret[f"mask_{m}"] = loss_loc.item()

        return loss, loss_ret
    


## TAKEN from nnUNet

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp
def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn
class nnUNetSoftDiceLoss(torch.nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        if apply_nonlin is not None:
            self.apply_nonlin = eval(apply_nonlin)
        else:
            self.apply_nonlin = None
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        #x = einops.rearrange(x, "b h w c -> b c h w")
        #y = einops.rearrange(y, "b h w c -> b c h w")
        # x: BCHW, y: BCHW
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        with torch.no_grad():
            loss_ret = {}
            if self.batch_dice:
                for _class in range(dc.shape[0]):
                    loss_ret[f"mask_{_class}"] = 1 - dc[_class].item()
            else:
                for _class in range(dc.shape[1]):
                    loss_ret[f"mask_{_class}"] = 1 - dc[:, _class].mean().item()

        return -dc, loss_ret
