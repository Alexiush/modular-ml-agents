from setuptools import setup
from mlagents.plugins import ML_AGENTS_TRAINER_TYPE

setup(
    name="custom_ppo_plugin",
    version="0.0.1",

    entry_points={
        ML_AGENTS_TRAINER_TYPE: [
            "custom_ppo=custom_ppo_plugin.trainer:get_type_and_setting",
        ]
    },
)