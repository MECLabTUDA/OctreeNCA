
from matplotlib import pyplot as plt
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
import torch, math, numpy as np, einops
import torchio as tio
from src.utils.helper import merge_img_label_gt_simplified


class MedNCAAgent_extrapolation(MedNCAAgent):
    def initialize(self):
        super().initialize()
        assert self.exp.config['experiment.task'] == 'extrapolation', "Task must be extrapolation"

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        # data["image"]: BCHW
        # data["label"]: BCHW
        out = self.get_outputs(data)
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}
        #print(outputs.shape, targets.shape)
        #2D: outputs: BHWC, targets: BHWC
        loss, loss_ret = loss_f(**out)

        if loss != 0:
            loss.backward()

            if gradient_norm or self.exp.get_from_config('experiment.logging.track_gradient_norm'):
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

            if gradient_norm:
                max_norm = 1.0
                # Gradient normalization

                # Calculate scaling factor and scale gradients if necessary
                scale_factor = max_norm / (total_norm + 1e-6)
                if scale_factor < 1:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)

            if self.exp.get_from_config('experiment.logging.track_gradient_norm'):
                if not hasattr(self, 'epoch_grad_norm'):
                    self.epoch_grad_norm = []
                self.epoch_grad_norm.append(total_norm)

            self.optimizer.step()
            if not self.exp.get_from_config('trainer.update_lr_per_epoch'):
                self.update_lr()
            
            if self.exp.get_from_config('trainer.ema') and self.exp.get_from_config('trainer.ema.update_per') == 'batch':
                self.ema.update()

        return loss_ret

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> dict:
        
        inputs, targets = data['image'], data['label']
        del targets 
        targets = inputs.clone()
        #2D: inputs: BCHW, targets: BCHW
        margin = self.exp.get_from_config("experiment.task.margin")

        b,c,h,w = inputs.shape
        mask  = torch.zeros(b,1,h,w, device=inputs.device)

        if self.exp.config["experiment.task.direction"] == "top":
            mask[:, :, :margin, :] = 1
        elif self.exp.config["experiment.task.direction"] == "all":
            mask[:, :, :margin, :] = 1
            mask[:, :, -margin:, :] = 1
            mask[:, :, :, :margin] = 1
            mask[:, :, :, -margin:] = 1
        else:
            raise ValueError("Invalid direction")
        
        inputs = inputs * (1-mask)
        data['inputs'] = inputs


        out = self.model(inputs, targets, self.exp.get_from_config('trainer.batch_duplication'))

        #plt.subplot(1,2,1)
        #plt.imshow(einops.rearrange(inputs.cpu().numpy()[0], "c h w -> h w c"))
        #plt.subplot(1,2,2)
        #plt.imshow(out["pred"].cpu().numpy()[0])
        #plt.show()


        out['unpatched_target'] = einops.rearrange(targets, "b c h w -> b h w c")

        out['loss_mask'] = einops.rearrange(mask, "b c h w -> b h w c")

        #2D: inputs: BHWC, targets: BHWC
        return out
    
    @torch.no_grad()
    def test(self, loss_f: torch.nn.Module, save_img: list = None, tag: str = 'test/img/', 
             pseudo_ensemble: bool = False, split='test', ood_augmentation: tio.Transform | None=None,
             output_name: str|None=None) -> dict:
        loss_f = torch.nn.MSELoss()
        assert ood_augmentation is None, "OOD augmentation not supported for extrapolation"
        assert output_name is None, "Output name not supported for extrapolation"


        # Prepare dataset for testing
        dataset = self.exp.datasets[split]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        self.exp.set_model_state('test')

        # Prepare arrays
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0
        patient_real_Img = None
        loss_log = {}
        for m in range(self.exp.config['model.output_channels']):
            loss_log[m] = {}
        if save_img == None:
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]


        # For each data sample
        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data['id'], data['image'], data['label']
            if 'name' in data:
                name = data['name']
            out = self.get_outputs(data, full_img=True, tag="0")
            outputs, targets = out['pred'], out['unpatched_target']

            if isinstance(data_id, str):
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                text = str(data_id[0]).split('_')
                if len(text) == 3:
                    _, id, slice = text
                else:
                    id = data_id[0]
                    slice = None

            
            # --------------- 2D ---------------------
            # If next patient
            if (id != patient_id or dataset.slice == -1) and patient_id != None:
                out = str(patient_id) + ", "

                loss_log[0][patient_id] = loss_f(patient_3d_image, patient_3d_label).item()
                print("PATIENT ID", patient_id, loss_log[0][patient_id])

                
                patient_id, patient_3d_image, patient_3d_label = id, None, None
            # If first slice of volume
            if patient_3d_image == None:
                patient_id = id
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                patient_real_Img = inputs.detach().cpu()
            else:
                patient_3d_image = torch.vstack((patient_3d_image, outputs.detach().cpu()))
                patient_3d_label = torch.vstack((patient_3d_label, targets.detach().cpu()))
                patient_real_Img = torch.vstack((patient_real_Img, inputs.detach().cpu()))
            # Add image to tensorboard
            
            patient_real_Img = einops.rearrange(patient_real_Img, "b c h w -> b h w c")

            if i in save_img: 
                self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                                merge_img_label_gt_simplified(patient_real_Img, patient_3d_image, patient_3d_label, dataset.is_rgb, False),
                                self.exp.currentStep)
        # If 2D
        out = str(patient_id) + ", "
        for m in range(patient_3d_label.shape[-1]):
            if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() 
                out = out + str(loss_log[m][patient_id]) + ", "
            else:
                out = out + " , "
        print(out)
        # Print dice score per label
        for key in loss_log.keys():
            if len(loss_log[key]) > 0:
                average = sum(loss_log[key].values())/len(loss_log[key])
                print("Average Dice Loss 3d: " + str(key) + ", " + str(average))
                print("Standard Deviation 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))
                self.exp.write_scalar('Loss/test/' + str(key), average, self.exp.currentStep)
                self.exp.write_scalar('Loss/test_std/' + str(key), self.standard_deviation(loss_log[key]), self.exp.currentStep)

        self.exp.set_model_state('train')
        return loss_log