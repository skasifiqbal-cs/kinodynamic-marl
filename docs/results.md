# Getting a run's results to me

Training writes exactly one thing, on whichever machine trains:

```
runs/<shaping>_<network>_<obs>/<timestamp>/
    config.yaml               what it was trained with
    checkpoints/best_agent.pt the policy
    events.out.tfevents...    the curves
```

About 6 MB. That folder IS the result — everything below is just moving it.

`runs/` is gitignored apart from the 4 KB `config.yaml`: checkpoints are permanent once
committed, and a full run writes ~18 MB of them.

## Sending it (default: scp)

```bash
scp -r runs/<shaping>_<network>_<obs>/<timestamp> aditya@<ip>:~/inbox/
```

That is the whole protocol. No account, no upload, nothing to forget. Keep `config.yaml`
next to `checkpoints/` — that layout is what lets the receiver run

```bash
python evaluate.py eval.checkpoint=<timestamp>/checkpoints/best_agent.pt
```

with no other flags: env, shaping, obs and network all come from the saved config.

Curves, if wanted, come from the same folder:

```bash
tensorboard --logdir runs/
```

## W&B (optional, off by default)

Worth turning on for a long run you want to watch live from elsewhere, or when scp is not
possible — a machine off-campus, behind a firewall, or one you cannot reach by SSH.

```bash
pip install -e ".[dev,wandb]"
wandb login                                       # key from wandb.ai/authorize
echo 'export WANDB_ENTITY=<team>' >> ~/.bashrc    # else runs land in your personal account
python train.py wandb.enabled=true
```

`$WANDB_ENTITY` unset is the silent failure: nothing errors, training looks normal, and the
run is simply somewhere the rest of us cannot see. Check `echo $WANDB_ENTITY` first.

At the end, `config.yaml` + `best_agent.pt` are published as an artifact
(`src/approach/rl/train.py:upload_run_artifact`), which the receiver pulls with:

```bash
wandb artifact get $WANDB_ENTITY/kinodynamic-rl/<shaping>_<network>_<obs>_<timestamp>:latest --root /tmp/check
python evaluate.py eval.checkpoint=/tmp/check/checkpoints/best_agent.pt
```

The artifact name is the `runs/` path with the slash turned into an underscore — not the
`wandb/run-*` directory name. The `wandb/` directory in the repo root is the SDK's own
scratch space (debug logs plus an upload staging copy); it is gitignored and safe to delete.

Nothing is published if the run never wrote `best_agent.pt`, which happens when it is
shorter than one `checkpoint_interval`. Lower `train.checkpoint_interval` for test runs.

## Before you call a run done

Whichever transport you used, the receiver must be able to run those two lines and
reproduce your number. If they cannot, the run is not reportable — find out why first.
