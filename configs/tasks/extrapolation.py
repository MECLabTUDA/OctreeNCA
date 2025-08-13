extrapolation_task_config={
    'experiment.task': "extrapolation",
    'experiment.task.margin': 10, #remove 10 pixels from each border
    'experiment.task.score': ["torch.nn.L1Loss", "torch.nn.MSELoss"],
    'trainer.losses.parameters': [{}, {}],
    'experiment.task.direction': "all"
}
    