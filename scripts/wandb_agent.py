import wandb

if __name__ == "__main__":
    # Configure the sweep
    sweep_config = {
        'method': 'bayes',  # Search method
        'metric': {
            'name': 'val/metrics/mIoU',
            'goal': 'maximize'
        },
        'parameters': {
            'model.error_tolerance': {
                # 'distribution': 'uniform',
                'values': [0.5, 0.75, 1.0]
            },
            'model.orth_scale': {
                'values': [0.5, 0.75, 1.0]
            },
            'loss.center_weight':{
                'distribution': 'q_uniform',
                'min': 1.0,
                'max': 10.0,
                'q': 1.0
            },
            'loss.offset_weight':{
                'distribution': 'q_uniform',
                'min': 0.1,
                'max': 0.5,
                'q': 0.1
            }
        },
        'command':[ 
            '${env}',  # This makes sure the environment variables are properly set
            'python',
            'scripts/train_v2.py',
            '+experiment=GaussianLSS',
            'data.dataset_dir=/media/hcis-s20/SRL/nuscenes/trainval/',
            'data.labels_dir=/media/hcis-s20/SRL/nuscenes/trainval/cvt_labels_nuscenes_v4'
        ]
    }

    # Initialize the sweep in W&B
    sweep_id = wandb.sweep(sweep_config, project="cross_view_transformers_test")

    # Run the sweep
    wandb.agent(sweep_id, count=1)  # 'count' defines how many runs to execute
