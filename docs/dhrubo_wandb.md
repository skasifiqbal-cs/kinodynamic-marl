# Getting your runs to me

Your results have to reach me in a form I can check without asking you for files. That means
Weights & Biases, not the repo: checkpoints are permanent once committed and a full-length run
writes about 18 MB of them, so `runs/` stays ignored apart from the 4 KB `config.yaml`.

Do this once, before your first real run.

## Setup

Ask me for the invite to the `kinodynamic-rl` project and accept it — sign up with your
university address, the academic tier is free. Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,wandb]"
wandb login                       # paste the key from wandb.ai/authorize
```

The venv line is not optional, and this doc used to omit it. Installing into a conda `base`
environment puts torch, numpy and protobuf on top of whatever else lives there, which breaks
the other packages and can break `wandb login` outright: a stale `protobuf-*.egg` left in an
Anaconda `site-packages` gets prepended to `sys.path` by `easy-install.pth`, shadowing the
protobuf pip just installed, and its `google/__init__.py` calls `pkg_resources.declare_namespace`,
which newer setuptools removed. The traceback names the `.egg` path. A venv has none of that,
so make one rather than trying to repair the conda env.

Activate it in every new shell (`source .venv/bin/activate`) before running anything here.

`wandb login` writes the key to `~/.netrc`. It does not belong in the repo, in a config file, or
in a shell script you commit — if it ever lands in a commit, tell me and rotate it, don't just
delete the line.

Last piece, and the one that actually matters:

```bash
echo 'export WANDB_ENTITY=<the team name I send you>' >> ~/.bashrc && source ~/.bashrc
```

Without it your runs go to your personal account and I cannot see them. Nothing errors, nothing
warns — the training looks completely normal and the results are simply somewhere I have no
access to. Check it with `echo $WANDB_ENTITY` before you start a long run.

## Running

```bash
python train.py train.algorithm=mappo env=gap_2agent wandb.enabled=true
```

`wandb.enabled=true` is not optional. It stays off by default so the repo still works for people
without the extra installed, which means it is on you to add it every time.

Curves stream live. At the end, `config.yaml` and `best_agent.pt` are published as an artifact
(`src/approach/rl/train.py:upload_run_artifact`) — that is what lets me re-run your policy
instead of reading your numbers.

## Before you call a run done

Run these two lines yourself, from a different directory, and watch the GIF:

```bash
wandb artifact get $WANDB_ENTITY/kinodynamic-rl/<run_name>:latest --root /tmp/check
python evaluate.py eval.checkpoint=/tmp/check/checkpoints/best_agent.pt
```

No other flags: env, shaping, obs and network travel with the weights in the artifact. If those
two lines do not reproduce the number you are about to report, the run is not reportable — find
out why first. This is the same thing I will do, so you may as well hit the problem before I do.

Add `wandb.enabled=true` to the evaluate line and the GIF goes to the project too, which is the
quickest way to show me something without me downloading anything.

## When it goes wrong

`wandb: ERROR api_key not configured` — you skipped `wandb login`.

Run does not appear in the shared project — `$WANDB_ENTITY` is unset or misspelled. Look at the
URL the run prints: it names the account it actually went to.

`ModuleNotFoundError: wandb` — you installed `.[dev]` without `wandb`.

Artifact missing at the end of a training run — it only publishes if `best_agent.pt` exists, and
a run shorter than one `checkpoint_interval` never writes one. Lower
`train.checkpoint_interval` for short test runs.
