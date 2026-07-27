from __future__ import annotations

from scripts.play import _parse_rickshaw_options


def test_rickshaw_play_options_are_removed_before_mjlab_parsing() -> None:
    options, remaining = _parse_rickshaw_options(
        [
            "Mjlab-G1-Rickshaw-Slopes-Student",
            "--slope",
            "0.05",
            "--checkpoint-file",
            "model.pt",
        ]
    )

    assert options.slope == 0.05
    assert remaining == [
        "Mjlab-G1-Rickshaw-Slopes-Student",
        "--checkpoint-file",
        "model.pt",
    ]
