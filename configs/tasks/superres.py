superres_task_config = {
    'experiment.task': "super_resolution",
    'experiment.task.factor': 4,
    'experiment.task.train_on_residual': True,
    'experiment.task.score': ["torch.nn.L1Loss"],
    'trainer.losses': ["torch.nn.L1Loss"],
    'trainer.losses.parameters': [{}],
    'trainer.loss_weights': [1e2],
    }