from tests.envs import EnvFactory
import mujoco_py
import os
import Utils
import pandas as pd
import random
import math
import numpy as np
import cma
import skvideo


def is_successful_run(run_folder):
    output_file = os.path.join(run_folder, "output")
    if os.path.isfile(output_file):
        run_info = pd.read_csv(output_file, delimiter=",", header=0)
    return run_info["success"][0]


def is_unique(values):
    return (values - values[0] < 1e-6).all()


def collect_data_from_runs(env):

    # Get sub-folders
    runs = os.listdir(env.forward_dynamics_folder)

    # Go through all runs
    data = []
    for run in runs:

        # Check if this run was successful
        run_folder = os.path.join(env.forward_dynamics_folder, run)
        if not is_successful_run(run_folder):
            continue

        # Get states
        states, state_hdr = Utils.parse_sto_file(os.path.join(run_folder, "FDS_states.sto"))

        # Get controls
        controls, ctrl_hdr = Utils.parse_sto_file(os.path.join(run_folder, "controls.sto"))

        # Make sure OpenSim simulation was long enough (should be if it didn't fail, just double-checking)
        if states.index[-1] < controls.index[-2]:
            print("Did this run fail?\n  {}".format(run_folder))
            continue

        # We're interested only in a subset of states
        state_names = list(states)

        # Parse state names
        parsed_state_names = []
        for state_name in state_names:
            p = state_name.split("/")
            if p[-1] != "value":
                parsed_state_names.append(state_name)
            else:
                parsed_state_names.append(p[1])

        # Rename, filter, and reorder
        states.columns = parsed_state_names
        states = states.filter(items=env.target_states)
        states = states[env.target_states]

        # Reindex states with the same timestep control file uses
        timesteps = np.diff(controls.index.values)
        if not is_unique(timesteps):
            print("Check timesteps in the control file of run {}".format(run_folder))
            continue
        if timesteps[0] > 0.01:
            print("This timestep might be too big")
            continue

        states = Utils.reindex_dataframe(states, timesteps[0], last_timestamp=controls.index.values[-1])

        data.append({"states": states.values, "controls": controls.values,
                     "state_names": list(states), "muscle_names": list(controls),
                     "timestep": timesteps[0]})

    return data


def run_simulation(sim, model, controls, visualise=False, record=False):

    if visualise or record:
        viewer = mujoco_py.MjViewer(sim)
    if record:
        imgs = []

    qpos = np.empty((len(controls), len(model.joint_names)))

    # We assume there's one set of controls for each timestep
    for t in range(len(controls)):

        # Set muscle activations
        sim.data.ctrl[:] = controls[t, :]

        # Forward the simulation
        sim.step()

        # Get joint positions
        qpos[t, :] = sim.data.qpos

        if visualise:
            viewer.render()
        if record:
            imgs.append(np.flipud(sim.render(width=1600, height=1200, depth=False, camera_name="main")))

    if record:
        # Write the video
        writer = skvideo.io.FFmpegWriter("outputvideo.mp4", inputdict={"-s": "1600x1200", "-r": str(1/model.opt.timestep)})
        for img in imgs:
            writer.writeFrame(img)

        # Add some images so the video won't end abruptly
        for _ in range(50):
            writer.writeFrame(imgs[-1])

        writer.close()

    return qpos


def initialise_simulation(sim, model, timestep, initial_states=None):

    # Set timestep
    model.opt.timestep = timestep

    # Reset sim
    sim.reset()

    # Set initial states
    if initial_states is not None:
        if "qpos" in initial_states:
            sim.data.qpos[:] = initial_states["qpos"]
        if "qvel" in initial_states:
            sim.data.qvel[:] = initial_states["qvel"]
        if "ctrl" in initial_states:
            sim.data.ctrl[:] = initial_states["ctrl"]

        # We might need to call forward to make sure everything is set properly after setting qpos (not sure if required)
        sim.forward()


def do_optimization(env, data):

    # Initialise MuJoCo with the converted model
    model = mujoco_py.load_model_from_path(env.mujoco_model_file)

    # We need to set the location and orientation of the equality constraint to match the starting position of the
    # reference motion.
    #update_equality_constraint(model, kinematics_values)

    # Initialise simulation
    sim = mujoco_py.MjSim(model)

    # Make sure muscles are in the same order
    for run_idx in range(len(data)):
        if data[run_idx]["muscle_names"] != list(model.actuator_names):
            print("Muscles are in incorrect order, fix this")
            raise NotImplementedError

    # Get indices of target states
    target_state_indices = np.empty(len(env.target_states,), dtype=int)
    for idx, target_state in enumerate(env.target_states):
        target_state_indices[idx] = model.joint_names.index(target_state)

    # If there are initial states, reorder them according to model joints
    if env.initial_states is not None:
        qpos = np.zeros(len(model.joint_names),)
        qvel = np.zeros(len(model.joint_names),)
        ctrl = np.zeros(len(model.actuator_names),)

        if "joints" in env.initial_states:
            for state in env.initial_states["joints"]:
                idx = model.joint_names.index(state)
                if "qpos" in env.initial_states["joints"][state]:
                    qpos[idx] = env.initial_states["joints"][state]["qpos"]
                if "qvel" in env.initial_states["joints"][state]:
                    qvel[idx] = env.initial_states["joints"][state]["qvel"]

        if "actuators" in env.initial_states:
            for actuator in env.initial_states["actuators"]:
                idx = model.actuator_names.index(actuator)
                ctrl[idx] = env.initial_states["actuators"][actuator]

        initial_states = {"qpos": qpos, "qvel": qvel, "ctrl": ctrl}

    # Get initial values for params
    sigma = 100
    niter = 100
    params = [500] * len(model.actuator_names)
    opts = {"popsize": 8, "transformation": [lambda x: x ** 2, None], "tolfun": 1e-4,
            "CMA_diagonal": True, "maxiter": niter}
    optimizer = cma.CMAEvolutionStrategy(params, sigma, opts)

    while not optimizer.stop():
        solutions = optimizer.ask()

        fitness = []
        for solution in solutions:

            # Set proposed solutions to muscle scales
            for muscle in model.actuator_names:
                muscle_idx = model._actuator_name2id[muscle]
                model.actuator_gainprm[muscle_idx][3] = np.sqrt(solution[muscle_idx])

            # Test solution on all runs
            f = 0
            for run_idx in range(len(data)):
                states = data[run_idx]["states"]
                controls = data[run_idx]["controls"]
                timestep = data[run_idx]["timestep"]

                # Initialise sim
                initialise_simulation(sim, model, timestep, initial_states)

                # Run simulation
                qpos = run_simulation(sim, model, controls, visualise=False)

                # Calculate joint errors
                f += np.sum(Utils.estimate_joint_error(states, qpos[:, target_state_indices])) + 0.001*np.sum(solution)

            fitness.append(f)

        optimizer.tell(solutions, fitness)
        optimizer.logger.add()
        optimizer.disp()

    # Return found solution
    return np.sqrt(optimizer.result.xbest)

#    for muscle_idx in range(model.actuator_gainprm.shape[0]):
#        model.actuator_gainprm[muscle_idx][3] = np.sqrt(es.result.xbest[muscle_idx])
#    sim.reset()
#    output = run_simulation(sim, model, kinematics_values, control_values, visualise=False, record=True)

    # Compare joint values
#    estimate_joint_error(kinematics_values, output["qpos"], plot=True)

#    print(np.sqrt(es.result.xbest))
#    es.plot()


def main(model_name):

    # Get env
    env = EnvFactory.get(model_name)

    # Collect states and controls from successful runs
    data = collect_data_from_runs(env)

    # Shuffle the data and use first 80% of elements for training
    random.shuffle(data)
    cutoff_idx = math.ceil(0.8*len(data))
    train_set = data[:cutoff_idx]
    test_set = data[cutoff_idx:]

    # Do optimization with CMA-ES
    parameters = do_optimization(env, train_set)
    print(parameters)

    # Save parameters


if __name__ == "__main__":
    #main(sys.argv[1])
    main("mobl_arms")