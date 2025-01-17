from setuptools import setup
from mlagents.plugins import ML_AGENTS_TRAINER_TYPE

setup(
    name="modular_ml_agents",
    version="1.0.0",

    entry_points={
        ML_AGENTS_TRAINER_TYPE: [
            "custom_ppo=modular_ml_agents.custom_ppo_plugin.trainer:get_type_and_setting",
        ]
    },
)