cholec_dataset_config = {
    'experiment.dataset.img_path': r"<path>/data/cholecseg8k_preprocessed_2/",
    'experiment.dataset.label_path': r"<path>/data/cholecseg8k_preprocessed_2/",
    'experiment.dataset.keep_original_scale': True,
    'experiment.dataset.rescale': True,
    'experiment.dataset.input_size': [240, 432, 80],

    'experiment.dataset.split_file': r"<path>/octree_study/cholec_split.pkl", 

    'model.input_channels': 3,

    'trainer.num_steps_per_epoch': None,
    'trainer.batch_size': 2,
    'trainer.batch_duplication': 1,
}