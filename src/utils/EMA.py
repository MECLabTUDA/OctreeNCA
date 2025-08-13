import torch


class EMA():
    def __init__(self, model: torch.nn.Module, decay: float):
        """
        model  : the neural network model
        decay  : the decay rate for EMA
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.original_params = {}

        # Initialize shadow parameters with the model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().cpu()

    @torch.no_grad()
    def update(self):
        """Update the shadow parameters with the current model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data.cpu() + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    @torch.no_grad()
    def apply_shadow(self):
        """Apply the shadow parameters to the model and store the original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store the original parameters
                self.original_params[name] = param.data.clone().cpu()
                # Apply the shadow parameters
                param.data = self.shadow[name].to(param.data.device)

    @torch.no_grad()
    def restore_original(self):
        """Restore the original parameters to the model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Restore the original parameters
                param.data = self.original_params[name].to(param.data.device)

