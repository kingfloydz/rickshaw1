"""Runner regression for fixed startup domains."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import g1_rickshaw_lab.training_contract as contract
import g1_rickshaw_lab.rl.runner as runner_module
from g1_rickshaw_lab.rl.runner import RunnerContext, create_rickshaw_runner_type


class _FakeAlgorithm:
    def __init__(self, latent_dim: int = 16) -> None:
        self.actor = SimpleNamespace(latent_dim=latent_dim, history_length=61)
        self.update_calls = 0

    def update(self) -> int:
        self.update_calls += 1
        return self.update_calls

    def save(self) -> dict[str, Any]:
        return {}


class _FakeEnvironment:
    def __init__(self) -> None:
        self.global_reset_calls = 1
        self.common_step_counter = 0

    @property
    def unwrapped(self) -> _FakeEnvironment:
        return self


def _install_fake_runner(
    rollout_steps: int = 24,
) -> type:
    class FakeOnPolicyRunner:
        def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.env = env
            self.cfg = {
                "num_steps_per_env": rollout_steps,
                "save_interval": contract.training_artifact_interval(rollout_steps),
            }
            self.alg = _FakeAlgorithm()
            self.logger = SimpleNamespace(lenbuffer=[], save_model=lambda *args: None)
            self.current_learning_iteration = 0

        def learn(
            self, *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return args, kwargs

        def save(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def load(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            return {}

        def export_policy_to_jit(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def export_policy_to_onnx(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    configuration = {
        "training_parameters": {
            "rollout_steps": rollout_steps,
            "latent_dim": 16,
            "history_length": 61,
        }
    }
    context = RunnerContext(
        stage="s0_teacher",
        training_configuration=configuration,
        metadata=object(),
    )
    return create_rickshaw_runner_type(context, base_runner_type=FakeOnPolicyRunner)


def test_runner_never_resamples_or_resets_fixed_startup_domain(
) -> None:
    runner_type = _install_fake_runner()
    env = _FakeEnvironment()
    runner = runner_type(env)

    for _ in range(600):
        runner.alg.update()

    assert env.global_reset_calls == 1
    assert runner._g1_curriculum_iteration == 600


def test_runner_checkpoint_preserves_environment_step_counter(monkeypatch) -> None:
    saved_checkpoint: dict[str, Any] = {}
    monkeypatch.setattr(
        runner_module,
        "attach_checkpoint_metadata",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner_module,
        "atomic_torch_save",
        lambda checkpoint, path: saved_checkpoint.update(checkpoint),
    )
    runner_type = _install_fake_runner()
    env = _FakeEnvironment()
    env.common_step_counter = 12_345

    runner_type(env).save("checkpoint.pt", infos={"metric": 1})

    assert saved_checkpoint["infos"] == {
        "metric": 1,
        "env_state": {"common_step_counter": 12_345},
    }
