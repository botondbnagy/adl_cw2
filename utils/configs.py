configs = {
    "vit_4M_pretrain": {
        "image_size": 128,
        "patch_size": 16,
        "dim": 128,
        "depth": 12,
        "heads": 8,
        "mlp_dim": 512,
        "masking_ratio": 0.5,
        "lr": 1e-3,
        "weight_decay": 0.05,
        "batch_size": 64,
        "schedule_milestones": [50, 85],
        "schedule_gamma": 0.5,
        "epochs": 100,
        },
    "vit_4M_finetune": {
        "image_size": 128,
        "patch_size": 16,
        "dim": 128,
        "depth": 12,
        "heads": 8,
        "mlp_dim": 512,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "schedule_milestones": [70, 90, 95],
        "schedule_gamma": 0.5,
        "epochs": 100,
        "int_lr": 8e-4,
        "int_schedule_milestones": [180, 190],
        "int_schedule_gamma": 0.1,
        "int_epochs": 200,
        "mixup_alpha": 0.1,
        },
}