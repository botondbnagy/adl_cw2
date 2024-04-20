configs = {
    "vit_4M": {
        "image_size": 128,
        "patch_size": 16,
        "num_classes": 2, #not used
        "dim": 128,
        "depth": 12,
        "heads": 8,
        "mlp_dim": 512,
        "masking_ratio": 0.5,
        "lr": 1e-3,
        "weight_decay": 0.05,
        "schedule_milestones": [50, 85],
        "schedule_gamma": 0.5,
        "epochs": 100,
        },
}