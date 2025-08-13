octree = (221 /255, 30/255, 42/255)
m3d = (176 /255, 60/255, 168/255)
unet = (255 /255, 160/255, 0/255)
segformer = (119 /255, 151/255, 240/255)
transunet = (53 /255, 221/255, 77/255)
sam = (51 /255, 125/255, 86/255)

named_colors = {
    "OctreeNCA": octree,
    "M3D-NCA": m3d,
    "Med-NCA": m3d,

    "UNet": unet,
    "TransUNet": transunet,
    "SegFormer": segformer,
    "SAM": sam,

    "OctreeNCA-BN": unet,
    "OctreeNCA-LN": segformer,
}