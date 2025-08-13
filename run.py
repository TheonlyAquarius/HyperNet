import argparse
import yaml
from pathlib import Path
from src.diffusion.train_generator import train as train_gen
from src.train_target import train as train_tgt
from src.rollout import eval as roll

def entry():
    par = argparse.ArgumentParser()
    par.add_argument('--experiment_path', type=str, required=True, help='Path to the experiment directory.')
    args = par.parse_args()
    path_exp = Path(args.experiment_path)
    path_cfg = path_exp / 'config.yaml'
    if not path_cfg.is_file():
        print(f"Error: Configuration file not found at {path_cfg}")
        return
    with open(path_cfg, 'r') as fh:
        cfg = yaml.safe_load(fh)
    t = cfg.get('task')
    if t == 'train_generator':
        train_gen(cfg)
    elif t == 'train_target':
        train_tgt(cfg)
    elif t == 'rollout':
        r = roll(cfg)
        print(r)

if __name__ == '__main__':
    entry()
