"""Train an RL policy. One approach, one job — see main.py to dispatch on either.

    python train.py                                  # defaults: dijkstra shaping, mlp
    python train.py shaping=braking env=swap1_unicycle2
    python train.py train.timesteps=2000000 train.checkpoint_interval=100000

Writes to runs/{shaping}_{network}_{obs}/{timestamp}/, and saves the config there so
`python evaluate.py eval.checkpoint=<that dir>/checkpoints/agent_N.pt` needs no other
flags.
"""
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.approach.rl.train import run_training  # import after sys.path is set by hydra
    run_training(cfg)


if __name__ == "__main__":
    main()
