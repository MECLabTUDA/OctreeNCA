import torch, einops
from src.agents.Agent import BaseAgent
from src.agents.Agent_MedSeg2D import Agent_MedSeg2D
from src.agents.Agent_MedSeg3D import Agent_MedSeg3D
from src.datasets.Dataset_DAVIS import Dataset_DAVIS

class UNetAgent(Agent_MedSeg2D, Agent_MedSeg3D):
    """Base agent for training UNet models
    """

    def prepare_data(self, data: tuple, eval: bool = False) -> dict:
        r"""Prepare the data to be used with the model
            #Args
                data (int, tensor, tensor): identity, image, target mask
            #Returns:
                inputs (tensor): Input to model
                targets (tensor): Target of model
        """
        id, inputs, targets = data['id'], data['image'], data['label']
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        #2D: inputs: BHWC, targets: BHWC

        # for some reason other datasets come without designated channel dimension
        # hence, it is added here
        # Dataset_DAVIS returns RGB images with channel dimension so we dont need to do this here
        if self.exp.datasets['train'].slice is None and not self.exp.datasets['train'].delivers_channel_axis:
            inputs, targets = torch.unsqueeze(inputs, 1), targets #torch.unsqueeze(targets, 1) 
        
        #2D: inputs: BHWC, targets: BHWC
        
        if len(inputs.shape) == 4:
            inputs = inputs.permute(0, 3, 1, 2)
            targets = targets.permute(0, 3, 1, 2)
        
        #data = {'id': id, 'image': inputs, 'label': targets}
        data['image'] = inputs
        data['label'] = targets

        #2D: inputs: BCHW, targets: BCHW

        return data

    def get_outputs(self, data: tuple, **kwargs) -> dict:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        raise NotImplementedError("This method must be implemented in the child class")
        _, inputs, targets = data['id'], data['image'], data['label']
        if len(inputs.shape) == 4:
            return (self.model(inputs)).permute(0, 2, 3, 1), targets.permute(0, 2, 3, 1)
        else:
            return (self.model(inputs)).permute(0, 2, 3, 4, 1), targets #.permute(0, 2, 3, 4, 1)

    def prepare_image_for_display(self, image: torch.tensor) -> torch.Tensor:
        r"""Prepare image for display
            #Args
                image (tensor): image
        """
        return image
    
    def test(self, *args, **kwargs):
        dataset = self.exp.datasets["train"]
        if dataset.slice is not None:
            return Agent_MedSeg2D.test(self, *args, **kwargs)
        else:
            return Agent_MedSeg3D.test(self, *args, **kwargs)
