"""Run whichever approach the config selects, without caring which one it is.

    python main.py                                        # RL (default): train
    python main.py approach=planning approach.method=karc # planning: plan + evaluate

Use this when the approach is a variable — a sweep across RL and planning, or a script
that should not know the difference. When you already know which one you want, the
focused entry points below say so at the call site and are easier to read in history.

Four entry points, one job each:
    main.py              dispatch on `approach`: train (RL) or plan+evaluate (planning)
    train.py             train an RL policy
    evaluate.py          render an episode to GIF and report success
    scripts/fasteval.py  the same scoring, headless and in bulk — no rendering
"""
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.approach import build_approach  # after sys.path is set by hydra
    build_approach(cfg).run(cfg)


if __name__ == "__main__":
    main()
