# MAPPO

We already run IPPO through skrl. MAPPO is the same algorithm with one change: the critic sees
the global state instead of each agent's own observation. The actors don't change.

The environment side is already done, so don't go near `src/env/`. `state()` is at
`src/env/multiagent_nav.py:131` and returns the concatenated observations of all agents;
`state_spaces` is at `:128`; the vectorised path has the same at `src/env/vec_multiagent.py:95,47,125`.
skrl's trainer already calls `env.state()` every step and puts the result in `infos["shared_states"]`
(`skrl/trainers/torch/base.py:319-336`) — that code runs today under IPPO. Nothing to add there.

Everything you need to touch is in `src/approach/rl/train.py`.

Build the critic on the state space rather than the observation space:

```python
value_sp = env.state_space(agent_id) if algo == "mappo" else obs_sp
models[agent_id] = {
    "policy": build_policy(obs_sp, act_sp, device, cfg.network).to(device),
    "value":  build_value(value_sp, act_sp, device, cfg.network).to(device),
}
```

On a 2-agent env that takes the critic's input from 25 to 50. That line is the algorithm; miss it
and you get a shape mismatch at the first update.

Then swap the agent. Every hyperparameter key is identical between the two, so rename `ippo_cfg`
to `agent_cfg` and branch at the end:

```python
agent_cfg["shared_state_preprocessor"] = RunningStandardScaler
agent_cfg["shared_state_preprocessor_kwargs"] = {
    "size": env.state_space(env.possible_agents[0]), "device": device,
}
agent = MAPPO(
    possible_agents=env.possible_agents,
    models=models, memories=memories,
    observation_spaces={a: env.observation_space(a) for a in env.possible_agents},
    action_spaces={a: env.action_space(a) for a in env.possible_agents},
    shared_observation_spaces={a: env.state_space(a) for a in env.possible_agents},
    device=device, cfg=agent_cfg,
)
```

`shared_observation_spaces` sizes MAPPO's `shared_states` memory tensor
(`skrl/multi_agents/torch/mappo/mappo.py:238`). `shared_state_preprocessor` normalises the
critic's input — it is not `value_preprocessor`, which normalises the output and is set to
`false` deliberately in `conf/train/ppo_default.yaml:12`. Leave that one alone.

Add `algorithm: ippo` to `conf/train/ppo_default.yaml` and fold `algo` into `run_dir`, or the two
algorithms' runs land in the same directory. Then `python train.py train.algorithm=mappo`.

For checks: pytest and ruff clean, plus one test asserting the critic's first layer has
`n_agents * obs_dim` inputs — a run that merely starts proves nothing. Then run IPPO at the old
seed and confirm the reward curve matches what it did before your PR. That step is the point
rather than overhead: it is what catches a refactor that quietly broke the baseline. Finish with
a short run of each at the same seed, both curves on one plot.

Two things not to do. Don't add neighbour velocity to the observation. It is missing at
`src/obs/full_state.py:74-80` and it looks like an easy win, but it is exactly the information
MAPPO's critic is supposed to have and the actors are not — change both at once and the
comparison means nothing. Separate PR. And don't remove the IPPO path; it is the paper's baseline.

## Reporting runs

Train with W&B on, always:

```bash
python train.py train.algorithm=mappo env=gap_2agent wandb.enabled=true
```

Checkpoints stay out of git — they are permanent once committed and a full-length run writes
about 18 MB of them. What gets shared instead: the curves stream to W&B live, and at the end of
training the run's `config.yaml` and `best_agent.pt` are published as a W&B artifact
(`src/approach/rl/train.py:upload_run_artifact`). Only the 4 KB `config.yaml` is committed, so
the diff still records what you ran.

That artifact is what lets someone else check your result rather than take it on trust:

```bash
wandb artifact get kinodynamic-rl/<run_name>:latest --root /tmp/r
python evaluate.py eval.checkpoint=/tmp/r/checkpoints/best_agent.pt
```

No other flags — env, shaping, obs and network come from the config travelling with it. Run
those two lines yourself before saying a run is done; if they do not reproduce your number, the
run is not reportable.

To see the episode rather than the numbers, add `wandb.enabled=true` to the evaluate line and
the GIF lands in the same project.
