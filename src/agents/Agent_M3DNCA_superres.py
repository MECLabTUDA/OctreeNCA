import einops
import matplotlib.pyplot as plt
import torch
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.agents.Agent_UNet import UNetAgent
import torch.nn.functional as F
import random
from src.agents.Agent_MedSeg3D import Agent_MedSeg3D
from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2
from src.utils.MyDataParallel import MyDataParallel
from src.utils.helper import merge_img_label_gt_simplified
from src.utils.patchwise_inference import patchwise_inference3d
import torchio as tio
#import matplotlib
#matplotlib.use('Agg')
#start xming!!!


class M3DNCAAgent_superres(M3DNCAAgent):
    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> dict:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']

        #inputs: BCHWD
        #targets: BHWDC
        targets = inputs.clone()

        inputs = F.interpolate(inputs, scale_factor=1/self.exp.get_from_config('experiment.task.factor'), 
                               mode='trilinear')
        
        
        inputs = F.interpolate(inputs, size=targets.shape[2:], mode='trilinear')


        inputs = einops.rearrange(inputs, 'b c h w d -> b h w d c')
        targets = einops.rearrange(targets, 'b c h w d -> b h w d c')

        if self.exp.config['experiment.task.train_on_residual']:
            targets = targets - inputs

        visualize = False

        if visualize:
            fig = plt.figure()
            fig.add_subplot(1, 3, 1)
            plt.imshow(inputs[0, :, :, 12, 0].detach().cpu().numpy())
            fig.add_subplot(1, 3, 2)
            plt.imshow(targets[0, :, :, 12, 0].detach().cpu().numpy())
            print(F.mse_loss(inputs, targets))


        if self.exp.get_from_config('model.eval.patch_wise') and self.exp.model_state == 'test':
            inputs = patchwise_inference3d(inputs, self.model, self.exp.get_from_config('model.train.patch_sizes'))
            inputs = inputs[:, :, :, :,self.exp.get_from_config('model.input_channels'):self.exp.get_from_config('model.input_channels')+self.exp.get_from_config('model.output_channels')]
        else:
            out = self.model(inputs, targets, self.exp.get_from_config('trainer.batch_duplication'))

        if visualize:
            fig.add_subplot(1, 3, 3)
            plt.imshow(out["pred"][0, :, :, 12, 0].detach().cpu().numpy())
            plt.show()
            input("Press Enter to continue...")
        

        out["inputs"] = einops.rearrange(inputs, 'b h w d c -> b c h w d')
        if "target" not in out.keys():
            out["target"] = targets
        return out
    


    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        self.optimizer.zero_grad()
        data = self.prepare_data(data)
        rnd = random.randint(0, 1000000000)
        random.seed(rnd)
        out = self.get_outputs(data)
        #print("______________________")
        if self.exp.get_from_config('trainer.train_quality_control') in ["NQM", "MSE"]:
            raise NotImplementedError("NQM and MSE not implemented for 3D")
            random.seed(rnd)
            outputs2, targets2 = self.get_outputs(data)


        loss, loss_ret = loss_f(**out)


        if self.exp.get_from_config('trainer.train_quality_control') == "NQM":
            stack = torch.stack([outputs, outputs2], dim=0)
            outputs = torch.sigmoid(torch.mean(stack, dim=0))
            stack = torch.sigmoid(stack)
            if torch.sum(stack) != 0:
                mean = torch.sum(stack, axis=0) / stack.shape[0]
                stdd = torch.zeros(mean.shape).to(self.device)
                for id in range(stack.shape[0]):
                    img = stack[id] - mean
                    img = torch.pow(img, 2)
                    stdd = stdd + img
                stdd = stdd / stack.shape[0]
                stdd = torch.sqrt(stdd)

                print("STDD", torch.min(stdd), torch.max(stdd), torch.sum(outputs))

                if torch.min(stdd) > 0:
                    nqm = torch.sum(stdd) / torch.sum(outputs)

                    if nqm > 0:
                        print("NQM: ", nqm)
                        loss = loss + nqm #
        elif self.exp.get_from_config('trainer.train_quality_control') == "MSE":
            loss += F.mse_loss(outputs, outputs2)

            #print(nqm)

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
                print("GRADIENT NORM")
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
        return loss_ret
    

    @torch.no_grad()
    def test(self, loss_f: torch.nn.Module, save_img: list = None, tag: str = 'test/img/', 
             pseudo_ensemble: bool = False, 
             split='test', ood_augmentation: tio.Transform=None):
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        loss_f = torch.nn.MSELoss()

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
            save_img = []#1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        # For each data sample
        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            assert data['image'].shape[0] == 1, "Batch size must be 1 for evaluation"
            data_id, inputs, *_ = data['id'], data['image'], data['label']
            out = self.get_outputs(data, full_img=True, tag="0")

            
            outputs = out["pred"]
            inputs = out["inputs"]
            targets = out['target']

            if isinstance(data_id, str):
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                print("DATA_ID", data_id)
                text = str(data_id[0]).split('_')
                if len(text) == 3:
                    _, id, slice = text
                else:
                    id = data_id[0]
                    slice = None


            # Run inference 10 times to create a pseudo ensemble
            

            patient_3d_image = outputs.detach().cpu()
            patient_3d_label = targets.detach().cpu()
            patient_3d_real_Img = inputs.detach().cpu()
            patient_id = id
            #print(patient_id)

            loss_log[0][patient_id] = loss_f(patient_3d_image, patient_3d_label).item()

            print(",",loss_log[m][patient_id])
            # Add image to tensorboard
            if True: 
                if len(patient_3d_label.shape) == 4:
                    patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                middle_slice = int(patient_3d_real_Img.shape[3] /2)
                #print(patient_3d_real_Img.shape, patient_3d_image.shape, patient_3d_label.shape)
                self.exp.write_img(str(tag) + str(patient_id) + "_" + str(m),
                                merge_img_label_gt_simplified(patient_3d_real_Img, patient_3d_image, patient_3d_label, dataset.is_rgb, False), 
                                self.exp.currentStep)

                    
        # Print dice score per label
        for key in loss_log.keys():
            if len(loss_log[key]) > 0:
                print("Average Dice Loss 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))
                print("Standard Deviation 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))

        self.exp.set_model_state('train')
        return loss_log