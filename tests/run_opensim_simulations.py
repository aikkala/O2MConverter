import opensim
import os
import xmltodict
import sys
from timeit import default_timer as timer
from tests.envs import EnvFactory
import shutil
from time import sleep
import subprocess
import math
import pathlib
import numpy as np
import Utils


def forward_tool_process(setup_file, run_folder):

    # Initialise forward tool with the new setup file
    tool = opensim.ForwardTool(setup_file)

    # Run the tool
    start = timer()
    tool.run()
    end = timer()

    # Write output
    with open(os.path.join(run_folder, "output"), 'w') as f:
        f.write("success, run_time\n")
        f.write("1, {}\n".format(end-start))


def run_speed_test(env, runs, N):

    # Get timestep for OpenSim simulations
    if env.opensim_timestep is not None:
        timestep = env.opensim_timestep
    else:
        timestep = env.timestep

    # Loop through runs and simulate forward dynamics
    durations = np.zeros((N, len(runs)))
    for run_idx, run in enumerate(runs):

        # Get setup file
        run_folder = os.path.join(env.forward_dynamics_folder, run)
        setup_file = os.path.join(run_folder, "modified_setup_fd.xml")

        # Do each run 10 times
        for repeat_idx in range(N):

            # Initialise forward tool with the new setup file
            tool = opensim.ForwardTool(setup_file)

            # Get model
            model = tool.getModel()

            # Initialise model
            state = model.initSystem()

            # Load initial states
            storage = opensim.Storage(env.initial_states_file)
            ti = tool.getStartTime()
            tf = tool.getFinalTime() + timestep

            # Set initial states for controllers
            model.updControllerSet().setDesiredStates(storage)

            # Set initial states
            # Set initial states
            state_names = model.getStateVariableNames()
            for i in range(model.getNumStateVariables()):
                state_idx = storage.getStateIndex(state_names.get(i))
                if state_idx == -1:
                    state_idx = storage.getStateIndex('/jointset/' + state_names.get(i))
                if state_idx == -1:
                    state_idx = storage.getStateIndex('/forceset/' + state_names.get(i))
                if state_idx == -1:
                    raise IndexError
                model.setStateVariableValue(state, state_names.get(i),
                                            storage.getStateVector(0).getData().get(state_idx))

            # Open manager, set parameters
            manager = opensim.Manager(model)
            manager.setIntegratorAccuracy(integrator_accuracy)
            manager.setUseSpecifiedDT(True)
            L = math.ceil(tf / timestep)
            manager.setDTArray(opensim.Vector([timestep] * L))

            # Equilibrate muscles once initial state is set
            if equilibrate:
                model.equilibrateMuscles(state)

            # Integrate from ti to tf
            state.setTime(ti)
            manager.initialize(state)
            start = timer()
            # for i in range(1, L):
            #    manager.integrate(i*timestep)
            manager.integrate(tf)
            durations[repeat_idx, run_idx] = timer() - start

    return durations


def run_forward_tool(env, runs, N):

    # Loop through runs and simulate forward dynamics
    durations = np.zeros((N, len(runs)))
    for run_idx, run in enumerate(runs):

        # Get setup file
        run_folder = os.path.join(env.forward_dynamics_folder, run)
        setup_file = os.path.join(run_folder, "modified_setup_fd.xml")

        # Do each run 10 times
        for repeat_idx in range(N):

            # Initialise forward tool with the new setup file
            tool = opensim.ForwardTool(setup_file)

            # Run the tool
            start = timer()
            tool.run()
            durations[repeat_idx, run_idx] = timer() - start

        # Run forward tool as a separate process so we can kill it if it exceeds a time limit
        # (seems like the tool freezes/hangs if the simulation fails)
        #p = multiprocessing.Process(target=forward_tool_process, args=(modified_setup_xml, run_folder))
        #p.start()

        # Wait for 10 minutes, if the process is still running, kill it
        #p.join(5*60)
        #if p.is_alive():
        #    p.terminate()

            # Write output
        #    with open(os.path.join(run_folder, "output"), 'w') as f:
        #        f.write("success, run_time\n")
        #        f.write("0, 0\n")

    return durations


def run_forward_dynamics(env, runs, visualise=False):

    # Import Button, Controller only when actually using visualisations, this way we can run this code in Triton
    if visualise:
        from pynput.mouse import Button, Controller

    # We need to modify the setup file for each run
    with open(env.opensim_setup_file) as f:
        text = f.read()
    setup = xmltodict.parse(text)

    # Get timestep for OpenSim simulations
    if env.opensim_timestep is not None:
        timestep = env.opensim_timestep
    else:
        timestep = env.timestep

    # Loop through runs and simulate forward dynamics
    for run_idx, run in enumerate(runs):

        # Get forward dynamics folder
        run_folder = os.path.join(env.forward_dynamics_folder, run)
        print("Processing {}".format(run))

        # Skip this if FDS_states.sto file already exists
        if not visualise and os.path.isfile(os.path.join(run_folder, "FDS_states.sto")):
            continue
            #pass

        # Edit settings
        setup["OpenSimDocument"]["ForwardTool"]["@name"] = "tool"
        setup["OpenSimDocument"]["ForwardTool"]["model_file"] = env.opensim_model_file
        setup["OpenSimDocument"]["ForwardTool"]["ControllerSet"]["objects"]["ControlSetController"]["controls_file"] = os.path.join(run_folder, "controls.sto")
        setup["OpenSimDocument"]["ForwardTool"]["results_directory"] = "."
        setup["OpenSimDocument"]["ForwardTool"]["states_file"] = env.initial_states_file
        setup["OpenSimDocument"]["ForwardTool"]["minimum_integrator_step_size"] = timestep
        setup["OpenSimDocument"]["ForwardTool"]["maximum_integrator_step_size"] = timestep
        setup["OpenSimDocument"]["ForwardTool"]['solve_for_equilibrium_for_auxiliary_states'] = 'true' if equilibrate else 'false'
        setup["OpenSimDocument"]["ForwardTool"]["integrator_error_tolerance"] = integrator_accuracy

        # Save the modified setup file
        modified_setup_xml = os.path.join(run_folder, "modified_setup_fd.xml")
        with open(modified_setup_xml, 'w') as f:
            f.write(xmltodict.unparse(setup, pretty=True, indent="  "))

        # Load the model via forward tool so that controls are correctly loaded; note that we don't really
        # use the forward tool
        tool = opensim.ForwardTool(modified_setup_xml)

        # Get model
        model = tool.getModel()

        # Set some paths if we're using visualiser
        if visualise:

            model.setUseVisualizer(True)
            output_folder = os.path.join(env.output_folder, run)
            osim_video_folder = os.path.join(pathlib.Path().absolute(), "python3_1")
            output_video_file = os.path.join(output_folder, 'simulation.mp4')

            # Skip this run if the video already exists
            if os.path.isfile(output_video_file):
                continue
                #pass

            # Create the output_folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Remove osim_video_folder if it exists
            if os.path.exists(osim_video_folder):
                shutil.rmtree(osim_video_folder)

        # Initialise model
        state = model.initSystem()

        # Set visualiser params
        if visualise:
            visualizer = model.getVisualizer().getSimbodyVisualizer()
            visualizer.setCameraFieldOfView(0.785398)

            # Get camera position
            pos = env.camera_pos[:3]

            # Transform from OpenSim camera to MuJoCo camera (only rotations since position is the same)
            ref_x = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            ref_y = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            ref_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            R_om = np.matmul(np.matmul(ref_y, ref_x), ref_z).astype(np.float)

            # Rotate the camera
            rot_x = Utils.create_rotation_matrix([1, 0, 0], rad=env.camera_pos[3])
            rot_y = Utils.create_rotation_matrix([0, 1, 0], rad=env.camera_pos[4])
            rot_z = Utils.create_rotation_matrix([0, 0, 1], rad=env.camera_pos[5])
            rotation = np.matmul(np.matmul(np.matmul(R_om, rot_x[:3, :3]), rot_y[:3, :3]), rot_z[:3, :3])

            # Set camera transformation
            visualizer.setCameraTransform(opensim.Transform(opensim.Rotation(
                opensim.Mat33(*rotation.flatten())), opensim.Vec3(pos[0], pos[2], -pos[1])))

            visualizer.setBackgroundType(opensim.SimTKVisualizer.SolidColor)
            visualizer.setBackgroundColor(opensim.Vec3(0, 0, 0))
            visualizer.drawFrameNow(state)

            # Not sure how to change frame rate, hopefully this is the correct frame rate
            frame_rate = visualizer.getDesiredFrameRate()

        # Load initial states
        storage = opensim.Storage(env.initial_states_file)
        ti = tool.getStartTime()
        tf = tool.getFinalTime() + timestep

        # Set initial states for controllers
        model.updControllerSet().setDesiredStates(storage)

        # Set initial states
        state_names = model.getStateVariableNames()
        for i in range(model.getNumStateVariables()):
            state_idx = storage.getStateIndex(state_names.get(i))
            if state_idx == -1:
                state_idx = storage.getStateIndex('/jointset/' + state_names.get(i))
            if state_idx == -1:
                state_idx = storage.getStateIndex('/forceset/' + state_names.get(i))
            if state_idx == -1:
                raise IndexError
            model.setStateVariableValue(state, state_names.get(i), storage.getStateVector(0).getData().get(state_idx))

        # Open manager, set parameters
        manager = opensim.Manager(model)
        manager.setIntegratorAccuracy(integrator_accuracy)
        manager.setUseSpecifiedDT(True)
        L = math.ceil(tf/timestep)
        manager.setDTArray(opensim.Vector([timestep]*L))
        manager.setPerformAnalyses(True)

        # Equilibrate muscles once initial state is set
        if equilibrate:
            model.equilibrateMuscles(state)

        if visualise:
            # Initialise mouse for stupid automatic image record workaround
            mouse = Controller()

            # Record images of the simulation
            mouse.position = (3030, 700)
            mouse.click(Button.left, 1)
            sleep(1)
            mouse.position = (3100, 815)
            mouse.click(Button.left, 1)
            sleep(1)

        # Integrate from ti to tf
        analysis = model.getAnalysisSet().get(0)
        analysis.begin(state)
        state.setTime(ti)
        manager.initialize(state)
        try:
            for i in range(1, L):
                manager.integrate(i*timestep)
                analysis.step(manager.getState(), i)
        except:
            print(f"{run} failed")

        analysis.end(manager.getState())
        analysis.printResults("analysis", run_folder)

        if visualise:
            # Stop video capture
            mouse.position = (3030, 700)
            mouse.click(Button.left, 1)
            sleep(1)
            mouse.position = (3100, 815)
            mouse.click(Button.left, 1)

        # Save simulated states
        if visualise:
            output_file = os.path.join(output_folder, "FDS_states.sto")
        else:
            output_file = os.path.join(run_folder, "FDS_states.sto")
        manager.getStateStorage().printToFile(output_file, "w")

        if visualise:

            # Create the video from images; images should be in 'python3_1' folder
            if not os.path.exists(osim_video_folder):
                print("There's no output folder for run {}!".format(run))

            else:

                # Remove the video if it exists
                if os.path.isfile(output_video_file):
                    os.remove(output_video_file)

                # Create the video
                subprocess.call(["ffmpeg",
                                 "-r", str(frame_rate),
                                 "-i", os.path.join(osim_video_folder, "Frame%04d.png"),
                                 output_video_file])


# Set a couple of constants
integrator_accuracy = 5e-5
equilibrate = True


def main(model_name):

    # Get env
    env = EnvFactory.get(model_name)

    # Get runs
    runs = os.listdir(env.forward_dynamics_folder)

    # Process all
    run_forward_dynamics(env, runs)


if __name__ == "__main__":
    main(*sys.argv[1:])
