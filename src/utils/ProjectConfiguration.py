import os
class ProjectConfiguration:
    STUDY_PATH = r"<path>/octree_study_new/"
    VITB16_WEIGHTS = r"<path>/Documents/GitHub/PretrainedVITs/imagenet21k_R50+ViT-B_16.npz"
    FILER_BASE_PATH = r"<path>"

if not os.path.exists(ProjectConfiguration.FILER_BASE_PATH):
    ProjectConfiguration.FILER_BASE_PATH = r"<path>"