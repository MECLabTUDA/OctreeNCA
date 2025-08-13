default_config ={
    'experiment.data_split': [0.7, 0, 0.3],
    'experiment.save_interval': 50,
    'experiment.device': "cuda:0",

    'experiment.logging.also_eval_on_train': True,
    'experiment.logging.track_gradient_norm': True,
    'experiment.logging.evaluate_interval': 50,



    'performance.compile': False,
    'performance.data_parallel': False,
    'performance.num_workers': 8,
    'performance.unlock_CPU': True,
    'performance.inplace_operations': True,

    'trainer.datagen.batchgenerators': True,
    'trainer.datagen.augmentations': True,
    'trainer.datagen.difficulty_weighted_sampling': False,

    'trainer.gradient_accumulation': False,
    'trainer.train_quality_control': False, #or "NQM" or "MSE"
}