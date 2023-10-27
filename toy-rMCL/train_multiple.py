import os

seed = 2023
for i in range(1, 21):
    print(f"Training Run: {i}, Seed: {seed}")
    os.system(f"python train.py --override_config_path=config/override_config_single.yaml --seed={seed}")
    seed += 1