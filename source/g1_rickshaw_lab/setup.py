"""Installation script for the G1 rickshaw mjlab task."""

from setuptools import find_packages, setup

setup(
    name="g1_rickshaw_lab",
    version="0.2.0",
    description="MuJoCo/mjlab manager-based G1 rickshaw task",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10,<3.14",
    install_requires=[
        "torch>=2.7.0",
        "numpy<2.5",
        "mujoco==3.10.0",
        "mjlab==1.5.3",
        "mujoco-warp==3.10.0.3",
        "rsl-rl-lib==5.4.0",
        "scipy>=1.15.0",
        "PyYAML>=6.0",
        "tensorboard",
    ],
    zip_safe=False,
)
