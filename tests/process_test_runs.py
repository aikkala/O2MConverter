import pickle
from tests.envs import EnvFactory
import mujoco_py
import numpy as np
import Utils
import os
import tests.run_opensim_simulations
import sys


def calculate_joint_errors(env, viewer, sim, data, target_state_indices, initial_states=None, condition=None):

    # Go through each run in test_data, get joint errors and record a video
    for run_idx in range(len(data)):
        print(run_idx)

        # Get data for this run
        states = data[run_idx]["states"]
        controls = data[run_idx]["controls"]
        timestep = data[run_idx]["timestep"]
        run = data[run_idx]["run"]

        # Make sure output folder exists
        os.makedirs(os.path.join(env.output_folder, run), exist_ok=True)

        # Initialise sim
        Utils.initialise_simulation(sim, timestep, initial_states)

        # Run simulation
        qpos = Utils.run_simulation(
            sim, controls, viewer=viewer,
            output_video_file=os.path.join(env.output_folder, run, "{}.mp4".format(condition)))

        # Get timesteps (we assume there's one set of controls per timestep)
        timesteps = np.arange(timestep, (controls.shape[0]+1)*timestep, timestep)

        # Calculate joint errors
        run_err = Utils.estimate_joint_error(states, qpos[:, target_state_indices], plot=True,
                                             joint_names=env.target_states, timesteps=timesteps,
                                             output_file=os.path.join(env.output_folder, run,
                                                                      "{}.png".format(condition)),
                                             error="MAE")

        # Add joint errors to data
        if "errors" not in data[run_idx]:
            data[run_idx]["errors"] = {}
        data[run_idx]["errors"][condition] = run_err


def run_mujoco_simulations(env, params, test_data, output_folder):

    # Open MuJoCo model and initialise a simulation
    model = mujoco_py.load_model_from_path(env.mujoco_model_file)
    sim = mujoco_py.MjSim(model)

    # Check muscle order
    Utils.check_muscle_order(model, test_data)

    # Get indices of target states
    target_state_indices = Utils.get_target_state_indices(model, env)

    # Get initial states
    initial_states = Utils.get_initial_states(model, env)

    # Update camera position
    cam_id = model._camera_name2id["for_testing"]
    model.cam_pos[cam_id, :] = env.camera_pos[:3]
    model.cam_quat[cam_id, :] = env.camera_pos[3:]

    # Run accuracy analysis first with default parameters
    #viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)
    viewer = None
    calculate_joint_errors(env, viewer, sim, test_data, target_state_indices, initial_states, "default_parameters")

    # Set parameters and calculate errors again
    Utils.set_parameters(model, params["parameters"], params["muscle_idxs"], params["joint_idxs"])
    calculate_joint_errors(env, viewer, sim, test_data, target_state_indices, initial_states, "optimized_parameters")


def main(model_name):

    # Load test data
    env = EnvFactory.get(model_name)
    with open(env.data_file, 'rb') as f:
        params, data, train_idxs, test_idxs = pickle.load(f)

    # Create a folder for output figures if it doesn't exist
    output_folder = os.path.join(env.output_folder, '..', 'figures')
    os.makedirs(output_folder, exist_ok=True)

    # Get test data
    test_data = [data[idx] for idx in test_idxs]

    # Record MuJoCo videos and get joint errors
    run_mujoco_simulations(env, params, test_data, output_folder)

    # Save data again with default and optimized parameter errors
    Utils.save_data(env.data_file, [params, data, train_idxs, test_idxs])

    # Record opensim videos
    tests.run_opensim_simulations.run_forward_dynamics(env, [t["run"] for t in test_data], True)


if __name__ == "__main__":
    main(*sys.argv[1:])
