peso_dataset_config = {
        'experiment.dataset.img_path': r"PESO/peso_training",
        'experiment.dataset.label_path': r"PESO/peso_training",
        'experiment.dataset.keep_original_scale': True,
        'experiment.dataset.rescale': True,
        'experiment.dataset.input_size': [320, 320],
        'experiment.dataset.img_level': 1,
        'experiment.dataset.patches_path': r"<path>/data/PESO_patches/",

        'experiment.dataset.seed': 42,

        'model.input_channels': 3,

        'trainer.num_steps_per_epoch': 200,
        'trainer.batch_size': 3,
        'trainer.batch_duplication': 1,
    }