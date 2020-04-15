from tests.envs import EnvFactory
from tests.envs import OsimWrapper
import os
import Utils
from pynput.mouse import Button, Controller
import opensim
from time import sleep
import numpy as np
import shutil
import subprocess


def main(model_name):

    # Get env
    env = EnvFactory.get(model_name)

    # Get OsimWrapper
    wrapper = OsimWrapper(env, visualize=True)

    # Wait a sec for visualizer window to initialize
    sleep(2)

    # Initialise mouse for stupid automatic image record workaround
    mouse = Controller()

    # Set frame rate and camera parameters
    visualizer = wrapper.osim_model.model.getVisualizer().getSimbodyVisualizer()
    visualizer.setCameraFieldOfView(0.785398)
    pos = env.camera_pos[:3]
    visualizer.setCameraTransform(opensim.Transform(opensim.Rotation(opensim.Mat33(0, 0, 1, 0, 1, 0, -1, 0, 0)),
                                                    opensim.Vec3(pos[0], pos[2], -pos[1])))
    visualizer.setBackgroundType(opensim.SimTKVisualizer.SolidColor)
    visualizer.setBackgroundColor(opensim.Vec3(0, 0, 0))

    # Can't figure out how to change frame rate, let's use default (which is this hopefully)
    frame_rate = visualizer.getDesiredFrameRate()

    # Run simulation for a couple of frames to make sure background color has really changed
    wrapper.reset()
    for _ in range(5):
        wrapper.step([0]*len(env.initial_states["actuators"]))

    # Go through all runs and record videos of them
    runs = os.listdir(env.output_folder)

    # Some runs fail
    failing_runs = ["run_1586166189310"]

    for run in runs:

        # If outputs are already produces (states file and simulation video) don't process this subject
        run_folder = os.path.join(env.forward_dynamics_folder, run)
        states_output_file = os.path.join(run_folder, 'osim_states.sto')
        output_video_file = os.path.join(env.output_folder, run, 'simulation.mp4')
        if os.path.isfile(states_output_file) and os.path.isfile(output_video_file):
            continue

        # Some runs fail, don't process them
        if run in failing_runs:
            continue

        # Make sure 'python3_1' folder doesn't exist
        osim_video_folder = '/home/aleksi/Workspace/O2MConverter/tests/python3_1'
        if os.path.exists(osim_video_folder):
            shutil.rmtree(osim_video_folder)

        # Reset osim simulation
        wrapper.reset()

        # Get controls
        values, hdr = Utils.parse_sto_file(os.path.join(run_folder, 'controls.sto'))
        controls = values.values
        muscle_names = list(values)

        # Make sure muscles are in correct order
        if muscle_names != wrapper.muscle_names:
            raise IndexError('Muscles are in incorrect order')

        # A stupid workaround to record images of the simulation
        mouse.position = (3030, 700)
        mouse.click(Button.left, 1)
        sleep(1)
        mouse.position = (3100, 815)
        mouse.click(Button.left, 1)
        sleep(1)

        # Run simulation
        states = np.zeros((controls.shape[0], len(wrapper.joint_names)+1))
        for i in range(controls.shape[0]):
            obs, _, _, _ = wrapper.step(controls[i, :])
            states[i, 0] = wrapper.osim_model.get_state().getTime()
            states[i, 1:] = obs

        # Stop video capture
        mouse.position = (3030, 700)
        mouse.click(Button.left, 1)
        sleep(1)
        mouse.position = (3100, 815)
        mouse.click(Button.left, 1)

        # Save states for later analysis
        states_output_file = os.path.join(run_folder, 'osim_states.sto')

        # Delete the file if it exists
        if os.path.isfile(states_output_file):
            os.remove(states_output_file)

        # Create the file
        with open(states_output_file, 'w') as file:

            # Write 'endheader' so we can use Utils.parse_sto_file to read the file
            file.write('endheader\n')

            # Write header
            file.write('time\t' + '\t'.join(wrapper.joint_names) + '\n')

            # Write the array
            np.savetxt(file, states, delimiter='\t')

        # Create the video from images; images should be in 'python3_1' folder
        if not os.path.exists(osim_video_folder):
            print("There's no output folder for run {}!".format(run_folder))
        else:

            # Remove the video if it exists
            if os.path.isfile(output_video_file):
                os.remove(output_video_file)

            # Create the video
            subprocess.call(["ffmpeg",
                             "-r", str(frame_rate),
                             "-i", os.path.join(osim_video_folder, "Frame%04d.png"),
                             output_video_file])


if __name__ == "__main__":
    #main(sys.argv[1])
    main("mobl_arms")
