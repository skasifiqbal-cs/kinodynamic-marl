"""Policy networks: MLP and GRU variants."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from skrl.models.torch import GaussianMixin, Model

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def _mlp_stack(in_dim: int, hidden_sizes: list[int], activation: str) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), act_cls()]
        prev = h
    return nn.Sequential(*layers), prev


class MLPPolicy(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, network_cfg: DictConfig) -> None:
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True)
        hidden = list(network_cfg.hidden_sizes)
        act = network_cfg.get("activation", "tanh")
        self.net, last_dim = _mlp_stack(obs_space.shape[0], hidden, act)
        self.head = nn.Linear(last_dim, act_space.shape[0])
        # Action-space scale/bias so the tanh-bounded mean lands inside [low, high].
        low  = torch.as_tensor(act_space.low,  dtype=torch.float32, device=device)
        high = torch.as_tensor(act_space.high, dtype=torch.float32, device=device)
        self.register_buffer("_act_scale", (high - low) / 2.0)
        self.register_buffer("_act_bias",  (high + low) / 2.0)
        # Init std ≈ 0.3·range (log(0.3) ≈ -1.2) — far smaller than the old std=1.0.
        self.log_std = nn.Parameter(torch.full((act_space.shape[0],), -1.2))

    def compute(self, inputs, role):
        raw = self.head(self.net(inputs["states"]))
        mean = torch.tanh(raw) * self._act_scale + self._act_bias
        return mean, self.log_std, {}


class GRUPolicy(GaussianMixin, Model):
    """Recurrent policy: GRU → optional MLP head.

    Hidden state reset on episode termination is handled via `terminated` key in inputs.
    Requires SequenceMemory in SKRL trainer for proper training.
    """

    def __init__(self, obs_space, act_space, device, network_cfg: DictConfig) -> None:
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True)

        self.hidden_size = int(network_cfg.hidden_size)
        self.num_layers = int(network_cfg.get("num_layers", 1))
        self.sequence_length = int(network_cfg.get("sequence_length", 16))

        self.gru = nn.GRU(
            input_size=obs_space.shape[0],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        post_hidden = list(network_cfg.get("post_gru_hidden_sizes", []))
        act = network_cfg.get("activation", "tanh")
        if post_hidden:
            self.post, last_dim = _mlp_stack(self.hidden_size, post_hidden, act)
        else:
            self.post = nn.Identity()
            last_dim = self.hidden_size

        self.head = nn.Linear(last_dim, act_space.shape[0])
        low  = torch.as_tensor(act_space.low,  dtype=torch.float32, device=device)
        high = torch.as_tensor(act_space.high, dtype=torch.float32, device=device)
        self.register_buffer("_act_scale", (high - low) / 2.0)
        self.register_buffer("_act_bias",  (high + low) / 2.0)
        self.log_std = nn.Parameter(torch.full((act_space.shape[0],), -1.2))

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, 1, self.hidden_size)],
            }
        }

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        h0 = inputs.get("rnn", [None])[0]

        if h0 is not None and terminated is not None:
            h0 = h0 * (~terminated.bool()).view(1, -1, 1)

        if states.dim() == 2:
            states = states.unsqueeze(1)

        out, h_n = self.gru(states, h0)
        out = self.post(out[:, -1, :])
        mean = torch.tanh(self.head(out)) * self._act_scale + self._act_bias
        return mean, self.log_std, {"rnn": [h_n]}


def build_policy(obs_space, act_space, device, network_cfg: DictConfig) -> Model:
    t = network_cfg.type
    if t == "mlp":
        return MLPPolicy(obs_space, act_space, device, network_cfg)
    if t == "gru":
        return GRUPolicy(obs_space, act_space, device, network_cfg)
    raise ValueError(f"Unknown network type: {t!r}. Choose 'mlp' or 'gru'.")
