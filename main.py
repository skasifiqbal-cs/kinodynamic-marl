"""Unified entry point — dispatches on the selected approach.

    python main.py                                          # reinforcement_learning (default): train
    python main.py approach=planning approach.method=rrt    # planning: plan + evaluate

`train.py` / `evaluate.py` / `scripts/fasteval.py` remain as focused entry
points; this is the single dispatcher over every approach.
"""
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.approach import build_approach  # after sys.path is set by hydra
    build_approach(cfg).run(cfg)


if __name__ == "__main__":
    main()
