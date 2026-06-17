"""Value (critic) networks: MLP and GRU variants."""
from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from skrl.models.torch import Model, DeterministicMixin
from src.networks.policy import _mlp_stack


class MLPValue(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device, network_cfg: DictConfig) -> None:
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        hidden = list(network_cfg.hidden_sizes)
        act = network_cfg.get("activation", "tanh")
        self.net, last_dim = _mlp_stack(obs_space.shape[0], hidden, act)
        self.head = nn.Linear(last_dim, 1)

    def compute(self, inputs, role):
        return self.head(self.net(inputs["states"])), {}


class GRUValue(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device, network_cfg: DictConfig) -> None:
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.hidden_size = int(network_cfg.hidden_size)
        self.num_layers  = int(network_cfg.get("num_layers", 1))
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
        self.head = nn.Linear(last_dim, 1)

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
        return self.head(out), {"rnn": [h_n]}


def build_value(obs_space, act_space, device, network_cfg: DictConfig) -> Model:
    t = network_cfg.type
    if t == "mlp":
        return MLPValue(obs_space, act_space, device, network_cfg)
    if t == "gru":
        return GRUValue(obs_space, act_space, device, network_cfg)
    raise ValueError(f"Unknown network type: {t!r}.")
