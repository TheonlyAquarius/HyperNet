import argparse
import yaml
import os
from pathlib import Path

def main():
    """
    Main dispatcher function.
    """
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path to the experiment directory.')
    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)
    config_path = experiment_path / 'config.yaml'

    if not config_path.is_file():
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Loaded configuration:")
    print(yaml.dump(config, indent=2))

    # TODO: Dispatch to the appropriate function in src/ based on the config
    # For example:
    # if config.get('task') == 'train_generator':
    #     from src.diffusion.train_generator import train_generator
    #     train_generator(config)
    # elif config.get('task') == 'synthesize_weights':
    #     from src.diffusion.synthesize_weights import synthesize_weights
    #     synthesize_weights(config)

if __name__ == '__main__':
    main()
