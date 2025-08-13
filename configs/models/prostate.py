prostate_model_config = {
        'model.channel_n': 16,
        'model.fire_rate': 0.5,
        'model.kernel_size': [3, 3, 3, 3, 3],
        'model.hidden_size': 64,
        'model.batchnorm_track_running_stats': False,

        'model.train.patch_sizes': [[80, 80, 6], [80, 80, 6], None, None, None],
        'model.train.loss_weighted_patching': False,

        'model.eval.patch_wise': False,

        'model.octree.res_and_steps': [[[320,320,24], 5], [[160,160,12], 5], [[80,80,6], 5], [[40,40,6], 5], [[20,20,6], 20]],
        'model.octree.separate_models': True,
        'model.backbone_class': "BasicNCA3DFast",

        'model.vitca': False,
        'model.vitca.depth': 1,
        'model.vitca.heads': 4,
        'model.vitca.mlp_dim': 64,
        'model.vitca.dropout': 0.0,
        'model.vitca.positional_embedding': 'vit_handcrafted', #'vit_handcrafted', 'nerf_handcrafted', 'learned', or None for no positional encoding
        'model.vitca.embed_cells': True,
        'model.vitca.embed_dim': 128,
        'model.vitca.embed_dropout': 0.0,
        }