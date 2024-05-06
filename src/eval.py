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

from colosseum import ASSETS_CONFIGS_FOLDER, TASKS_TTM_FOLDER, TASKS_PY_FOLDER
from colosseum.rlbench.utils import ObservationConfigExt, check_and_make, name_to_class
from colosseum.rlbench.extensions.environment import EnvironmentExt

NUM_EPISODES = 5
SAVE_EPISODES = False
HEADLESS = False


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
    assert task_class is not None, f"Can't get task-class for task {cfg.env.task_name}"

    rlbench_env = EnvironmentExt(
        action_mode=action_mode,
        obs_config=ObservationConfigExt(cfg.data),
        headless=HEADLESS,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=cfg.env,
    )
    rlbench_env.launch()

    visualizer = ValueMapVisualizer(config["visualizer"])
    env = VoxPoserRLBench(
        env=rlbench_env,
        visualizer=visualizer,
        save_episodes=SAVE_EPISODES,
    )
    lmps, _ = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps["plan_ui"]
    env.load_task(task_class)

    assert env.task is not None

    path_task = os.path.join(cfg.data.save_path, cfg.env.task_name)
    if SAVE_EPISODES:
        check_and_make(path_task)

    for index in range(0, NUM_EPISODES):
        descriptions, _ = env.reset()

        set_lmp_objects(lmps, env.get_object_names())

        instruction = np.random.choice(descriptions)
        try:
            voxposer_ui(instruction)
            success, terminate = env.task._task.success()
            truncated = False
        except:
            success, terminate = False, True
            truncated = True

        path_save_episodes = os.path.join(path_task, f"episode{index}")

        if SAVE_EPISODES:
            check_and_make(path_save_episodes)
            env.task._recorder.save(path_save_episodes)
            # if env.recorder is not None:
            #     env.recorder.save(path_save_episodes)

        print(
            f"{index + 1}/{NUM_EPISODES} : success={success}, "
            + f"terminate={terminate}, truncated={truncated}"
        )

        with open(f"success_terminate_{cfg.env.task_name}.txt", "a") as fhandle:
            fhandle.write(
                f"Success: {success}, Terminate: {terminate}, "
                + f"Instruction: {instruction}, Truncated: {truncated}\n"
            )

    return 0


if __name__ == "__main__":
    SystemExit(main())
