nca_trainer_config = {
    'trainer.optimizer': "torch.optim.Adam",
    'trainer.optimizer.lr': 0.0016,
    'trainer.optimizer.betas': [0.9, 0.99],
    'trainer.lr_scheduler': "torch.optim.lr_scheduler.ExponentialLR",
    'trainer.lr_scheduler.gamma': 0.9999**8,
    'trainer.update_lr_per_epoch': True,
    'trainer.normalize_gradients': None, # all, layerwise, none

    'trainer.n_epochs': 2000,

    'trainer.find_best_model_on': None,
    'trainer.always_eval_in_last_epochs': None,

    'trainer.ema': True,
    'trainer.ema.decay': 0.99,
    'trainer.ema.update_per': "epoch",
}