import os

import hydra
import numpy as np
import openai
from omegaconf import DictConfig

from arguments import get_config
from envs.colosseum_env import VoxPoserRLBench, CustomMoveArmThenGripper
from interfaces import setup_LMP
from utils import set_lmp_objects
from visualizers import ValueMapVisualizer

from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig

from colosseum import ASSETS_CONFIGS_FOLDER, TASKS_TTM_FOLDER, TASKS_PY_FOLDER
from colosseum.rlbench.utils import ObservationConfigExt, name_to_class
from colosseum.rlbench.extensions.environment import EnvironmentExt


@hydra.main(
    config_path=ASSETS_CONFIGS_FOLDER,
    config_name="slide_block_to_target.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> int:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    config = get_config("rlbench")

    action_mode = CustomMoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )

    task_class = name_to_class(cfg.env.task_name, TASKS_PY_FOLDER)
    assert (
        task_class is not None
    ), f"Can't get task-class for task {config.env.task_name}"

    rlbench_env = EnvironmentExt(
        action_mode=action_mode,
        obs_config=ObservationConfigExt(cfg.data),
        headless=False,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=cfg.env,
    )
    rlbench_env.launch()

    visualizer = ValueMapVisualizer(config["visualizer"])
    env = VoxPoserRLBench(rlbench_env, visualizer)
    lmps, _ = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps["plan_ui"]
    env.load_task(task_class)

    assert env.task is not None

    for index in range(0, 5):
        descriptions, _ = env.reset()

        set_lmp_objects(lmps, env.get_object_names())

        instruction = np.random.choice(descriptions)
        try:
            voxposer_ui(instruction)
            success, terminate = env.task._task.success()
        except:
            success, terminate = False, True

        print(f"{index + 1}/5 : success={success}, terminate={terminate}")

    return 0


if __name__ == "__main__":
    SystemExit(main())
