from __future__ import annotations

from scripts.train import _parse_rickshaw_options


def test_velocity_curriculum_option_is_removed_before_mjlab_parsing() -> None:
    options, remaining = _parse_rickshaw_options(
        [
            "Mjlab-G1-Rickshaw-Slopes-Teacher",
            "--velocity-curriculum",
            "--env.scene.num-envs",
            "8192",
        ]
    )

    assert options.velocity_curriculum
    assert remaining == [
        "Mjlab-G1-Rickshaw-Slopes-Teacher",
        "--env.scene.num-envs",
        "8192",
    ]
