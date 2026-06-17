"""Entry point.

Usage:
    python train.py                     # euclidean shaping (default)
    python train.py shaping=none
    python train.py shaping=dubins
    python train.py shaping=dubins train.timesteps=2000000
"""
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.training.train import run_training  # import after sys.path is set by hydra
    run_training(cfg)


if __name__ == "__main__":
    main()
