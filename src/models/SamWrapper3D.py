from collections import OrderedDict
import torch, einops
import torch.nn as nn
import matplotlib.pyplot as plt
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as np
import torchvision

class SamWrapper3D():
    def __init__(self, predictor: SAM2VideoPredictor):
        super().__init__()
        print(type(predictor))
        self.predictor = predictor

    def to(self, device):
        return self
    
    def parameters(self):
        return self.predictor.parameters()
    
    def train(self):
        return self
    
    def eval(self):
        return self
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor=None, batch_duplication: int=1):
        return self.forward(x, y, batch_duplication)



    def init_state_from_tensor(self, images, video_height, video_width,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,):
        """Initialize an inference state."""
        compute_device = self.predictor.device  # device of the model
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

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
    
    def sample_index(self, p):
        #https://stackoverflow.com/questions/61047932/numpy-sampling-from-a-2d-numpy-array-of-probabilities
        p = p / np.sum(p)
        i = np.random.choice(np.arange(p.size), p=p.ravel())
        return np.unravel_index(i, p.shape)

    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def put_annotations(self, inference_state, seg_class, seg, img_height, img_width):
        if seg.sum() == 0:
            print("no annotations for class", seg_class)
            return
        
        for ann_frame_idx in range(seg.shape[-1]):
            if seg[..., ann_frame_idx].sum() != 0:
                break

        print("putting annotations for class", seg_class, "on frame", ann_frame_idx)
        #ann_frame_idx = 0  # the frame index we interact with
        # seg.shape: (H, W, T)
        seg = seg[..., ann_frame_idx]
        seg = seg.cpu().numpy()


        input_prompt_points = []
        while len(input_prompt_points) < 1:
            x,y = self.sample_index(seg)
            input_prompt_points.append((y, x))

        input_prompt_points = np.array(input_prompt_points)
        input_prompt_point_labels = np.ones(len(input_prompt_points))

        # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=seg_class,
            points=input_prompt_points,
            labels=input_prompt_point_labels,
        )

        #plt.subplot(1, 2, 1)
        #for i, out_obj_id in enumerate(out_obj_ids):
        #    self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)



    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor=None, batch_duplication: int=1):
        #print(x.shape)  #torch.Size([1, 240, 432, 80, 3])
        #print(y.shape)  #torch.Size([1, 240, 432, 80, 5])
        #input("Press Enter to continue...")
        assert y is not None

        x = einops.rearrange(x, '1 h w d c -> d c h w')
        _,_,h,w = x.shape
        
        #resize to self.predictor.image_size
        x = torchvision.transforms.Resize((self.predictor.image_size,self.predictor.image_size))(x)
        

        #self.predictor.init_state()
        inference_state = self.init_state_from_tensor(x, h, w)

        for c in range(y.shape[4]):
            self.put_annotations(inference_state, c, y[0, :, :, :, c], h, w)
            #plt.subplot(1, 2, 2)
            #plt.imshow(y[0, :, :, 0, c].cpu().numpy())
            #plt.show()


        out_logits = -1e4 * torch.ones((y.shape[3], h, w, y.shape[4]), device=x.device)

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            for out_class_logit_idx, out_class in enumerate(out_obj_ids):
                #print(out_class, out_mask_logits[out_class].shape) #torch.Size([1, 240, 432])
                out_logits[out_frame_idx, :, :, out_class] = out_mask_logits[out_class_logit_idx][0]

        #out_logits.shape
        #input("Press Enter to continue...")

        out_logits = einops.rearrange(out_logits, 'd h w c -> 1 h w d c')

        self.predictor.reset_state(inference_state)
        return {#"pred": out_segmentation.to(in_device), 
                "logits": out_logits,
                }