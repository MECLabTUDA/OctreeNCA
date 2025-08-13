import os, shutil
from nnunet.dataset_conversion.utils import generate_dataset_json
import SimpleITK as sitk

INPUT_PATH = "<path>/nnUnet_raw_data_base/Prostate_MEDSeg"
OUT_PATH = "<path>/nnUnet_raw_data_base/nnUNet_raw_data/Task507_Prostate_MEDSeg"



os.makedirs(os.path.join(OUT_PATH, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(OUT_PATH, "labelsTr"), exist_ok=True)


cases_tr = []
for f in os.listdir(os.path.join(INPUT_PATH, "imagesTr")):
    name = f[:-len(".nii.gz")]
    cases_tr.append(name)

    in_path = os.path.join(INPUT_PATH, "imagesTr", f)
    out_path = os.path.join(OUT_PATH, "imagesTr", f"{name}_0000.nii.gz")
    image = sitk.ReadImage(in_path)
    image_arr = sitk.GetArrayFromImage(image)[0:1]
    image_new = sitk.GetImageFromArray(image_arr, isVector=False)
    dir = image.GetDirection()
    dir_new = (dir[0], dir[1], dir[2], dir[4], dir[5], dir[6], dir[8], dir[9], dir[10])
    image_new.SetOrigin(image.GetOrigin())
    image_new.SetSpacing(image.GetSpacing())
    #image_new.SetDirection(dir_new)
    image_new.SetDirection(dir)
    sitk.WriteImage(image_new, out_path)

for f in os.listdir(os.path.join(INPUT_PATH, "labelsTr")):
    in_seg = os.path.join(INPUT_PATH, "labelsTr", f)
    in_seg = sitk.ReadImage(in_seg)
    in_seg_arr = sitk.GetArrayFromImage(in_seg)
    in_seg_arr[in_seg_arr > 0] = 1
    out_seg = sitk.GetImageFromArray(in_seg_arr)
    out_seg.CopyInformation(in_seg)
    out_path = os.path.join(OUT_PATH, "labelsTr", f)
    sitk.WriteImage(out_seg, out_path)

generate_dataset_json(
    os.path.join(OUT_PATH, "dataset.json"),
    os.path.join(OUT_PATH, "imagesTr"),
    os.path.join(OUT_PATH, "labelsTr"),
    ('T1',),
    {0: 'background', 1: 'foreground' },
    "Prostate_MEDSeg"
)

#nnUNet_plan_and_preprocess -t 502