vitca_trainer_config = {
    'trainer.optimizer': "torch.optim.AdamW",
    'trainer.optimizer.lr': 1e-3,
    'trainer.lr_scheduler': "torch.optim.lr_scheduler.CosineAnnealingLR",
    'trainer.lr_scheduler.T_max': 2000,
    'trainer.update_lr_per_epoch': True,
    'trainer.normalize_gradients': "layerwise", # all, layerwise, none

    'trainer.n_epochs': 2000,

    'trainer.find_best_model_on': None,
    'trainer.always_eval_in_last_epochs': None,

    'trainer.ema': False,
    'trainer.ema.decay': 0.99,
    'trainer.ema.update_per': "epoch",
}