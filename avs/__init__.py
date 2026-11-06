"""Active Viewpoint Selection (AVS) for face perception under specular glare.

Core modules here are simulator-independent: they never import Isaac Sim, so
data processing, training and policy inference run on any machine. The Isaac
Sim adapter lives in ``avs.envs.isaac_env`` and imports Isaac lazily.
"""

__version__ = "0.1.0"
