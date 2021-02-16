import Utils
import numpy as np
#from osim.env.osim import OsimEnv
import pathlib
import os


class EnvFactory:

    class EnvTemplate:
        def __init__(self, model_name, timestep, opensim_setup_file, forward_dynamics_folder, mujoco_model_file, data_file,
                     params_file, output_folder, camera_pos, opensim_model_file, initial_states_file, target_states,
                     param_optim_pop_size, control_optim_pop_size, control_optim_diag, opensim_timestep=None):

            # Get project path
            self.project_path = pathlib.Path(__file__).parent.parent.absolute()

            self.model_name = model_name

            # Set parameters
            self.timestep = timestep
            self.opensim_setup_file = os.path.join(self.project_path, opensim_setup_file)
            self.forward_dynamics_folder = os.path.join(self.project_path, forward_dynamics_folder)
            self.mujoco_model_file = os.path.join(self.project_path, mujoco_model_file)
            self.data_file = os.path.join(self.project_path, data_file)
            self.params_file = os.path.join(self.project_path, params_file)
            self.output_folder = os.path.join(self.project_path, output_folder)
            self.camera_pos = camera_pos
            self.opensim_model_file = os.path.join(self.project_path, opensim_model_file)
            self.target_states = target_states
            self.param_optim_pop_size = param_optim_pop_size
            self.control_optim_pop_size = control_optim_pop_size
            self.control_optim_diag = control_optim_diag
            self.opensim_timestep = opensim_timestep

            # Read initial states from a file if given
            self.initial_states_file = os.path.join(self.project_path, initial_states_file)
            self.initial_states = {"joints": {}, "actuators": {}}
            states, hdr = Utils.parse_sto_file(self.initial_states_file)
            state_names = list(states)
            for state_name in state_names:
                if state_name.endswith(".fiber_length"):
                    continue
                elif state_name.endswith("_u"):
                    if state_name[:-2] not in self.initial_states["joints"]:
                        self.initial_states[state_name[:-2]] = {}
                    self.initial_states["joints"][state_name[:-2]]["qvel"] = states[state_name][0]
                elif state_name.endswith(".activation"):
                    self.initial_states["actuators"][state_name[:-11]] = states[state_name][0]
                elif '/' in state_name:
                    split = state_name.split('/')[1:]
                    if split[0] == "jointset":
                        if split[2] not in self.initial_states["joints"]:
                            self.initial_states["joints"][split[2]] = {}
                        if split[3] == "value":
                            self.initial_states["joints"][split[2]]["qpos"] = states[state_name][0]
                        elif split[3] == "speed":
                            self.initial_states["joints"][split[2]]["qvel"] = states[state_name][0]
                    elif split[0] == "forceset" and split[2] == "activation":
                        self.initial_states["actuators"][split[1]] = states[state_name][0]
                else:
                    if state_name not in self.initial_states["joints"]:
                        self.initial_states["joints"][state_name] = {}
                    self.initial_states["joints"][state_name]["qpos"] = states[state_name][0]

    mobl_arms = EnvTemplate("mobl_arms",
        0.002,
        'models/opensim/MoBL_ARMS_OpenSim_tutorial_33/setup_fd.xml',
        'tests/mobl_arms/forward_dynamics',
        'models/converted/MoBL_ARMS_model_for_testing_mujoco_converted/MoBL_ARMS_model_for_testing_mujoco_converted.xml',
        'tests/mobl_arms/output/data.pckl',
        'tests/mobl_arms/output/params.pckl',
        'tests/mobl_arms/output/simulations',
        np.array([1.8, -0.1, 0.7, np.pi/2, np.pi/2, 0]),
        'models/opensim/MoBL_ARMS_OpenSim_tutorial_33/MoBL_ARMS_model_for_testing_opensim.osim',
        'models/opensim/MoBL_ARMS_OpenSim_tutorial_33/initial_states.sto',
        ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"], 32, 32, True
    )

    MoBL_ARMS_no_wrap = EnvTemplate("mobl_arms_no_wrap",
        0.002,
        'models/opensim/MoBL_ARMS_OpenSim_tutorial_33/setup_fd.xml',
        'tests/mobl_arms_no_wrap/forward_dynamics',
        'models/converted/MoBL_ARMS_model_for_testing_mujoco_converted/MoBL_ARMS_model_for_testing_mujoco_converted.xml',
        'tests/mobl_arms_no_wrap/output/data.pckl',
        'tests/mobl_arms_no_wrap/output/params.pckl',
        'tests/mobl_arms_no_wrap/output/simulations',
        np.array([1.8, -0.1, 0.7, np.pi/2, np.pi/2, 0]),
        'models/opensim/MoBL_ARMS_OpenSim_tutorial_33/MoBL_ARMS_model_for_testing_opensim_no_wrapobjects.osim',
        'models/opensim/MoBL_ARMS_OpenSim_tutorial_33/initial_states.sto',
        ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"], 16, 32, True
    )

    gait2392_leg_dof = ["hip_flexion_", "hip_adduction_", "hip_rotation_", "knee_angle_", "ankle_angle_", "subtalar_angle_", "mtp_angle_"]
    gait2392 = EnvTemplate("gait2392",
        0.002,
        'models/opensim/Gait2392_Simbody/setup_fd.xml',
        'tests/gait2392/forward_dynamics',
        'models/converted/gait2392_millard2012muscle_for_testing_converted/gait2392_millard2012muscle_for_testing_converted.xml',
        'tests/gait2392/output/data.pckl',
        'tests/gait2392/output/params.pckl',
        'tests/gait2392/output/simulations',
        np.array([2, -2, 1, np.pi/2, np.pi/4, 0]),
        'models/opensim/Gait2392_Simbody/gait2392_millard2012muscle_for_testing.osim',
        'models/opensim/Gait2392_Simbody/initial_states.sto',
        [dof + "r" for dof in gait2392_leg_dof] + [dof + "l" for dof in gait2392_leg_dof] + ["lumbar_extension", "lumbar_bending", "lumbar_rotation"],
        32, 64, False
    )

    gait10dof18musc = EnvTemplate("gait10dof18musc",
        0.002,
        'models/opensim/Gait10dof18musc/setup_fd.xml',
        'tests/gait10dof18musc/forward_dynamics',
        'models/converted/gait10dof18musc_for_testing_converted/gait10dof18musc_for_testing_converted.xml',
        'tests/gait10dof18musc/output/data.pckl',
        'tests/gait10dof18musc/output/params.pckl',
        'tests/gait10dof18musc/output/simulations',
        np.array([2, -2, 1, np.pi/2, np.pi/4, 0]),
        'models/opensim/Gait10dof18musc/gait10dof18musc_for_testing.osim',
        'models/opensim/Gait10dof18musc/initial_states.sto',
        ["hip_flexion_r", "knee_angle_r", "ankle_angle_r", "hip_flexion_l", "knee_angle_l", "ankle_angle_l"],
        16, 32, True)

    rajagopal_leg_dof = ["hip_flexion_", "hip_adduction_", "hip_rotation_", "knee_angle_", "ankle_angle_",
                         "subtalar_angle_", "mtp_angle_"]
    rajagopal_arm_dof = ["arm_flex_", "arm_add_", "arm_rot_", "elbow_flex_", "pro_sup_", "wrist_flex_", "wrist_dev_"]
    rajagopal_walk = EnvTemplate("rajagopal_walk",
        0.002,
        'models/opensim/rajagopal_walking/setup_fd_walk.xml',
        'tests/rajagopal_walk/forward_dynamics',
        'models/converted/subject_scaled_walk_for_testing_converted/subject_scaled_walk_for_testing_converted.xml',
        'tests/rajagopal_walk/output/data.pckl',
        'tests/rajagopal_walk/output/params.pckl',
        'tests/rajagopal_walk/output/simulations',
        np.array([2, -2, 1, np.pi/2, np.pi/4, 0]),
        'models/opensim/rajagopal_walking/subject_scaled_walk_for_testing.osim',
        'models/opensim/rajagopal_walking/initial_states_walk.sto',
        [dof + "r" for dof in rajagopal_leg_dof] + [dof + "l" for dof in rajagopal_leg_dof] +
        [dof + "r" for dof in rajagopal_arm_dof] + [dof + "l" for dof in rajagopal_arm_dof] +
        ["lumbar_extension", "lumbar_bending", "lumbar_rotation"],
        1000, 64, False, opensim_timestep=0.0002)

    @staticmethod
    def get(env_name):
        assert hasattr(EnvFactory, env_name), f"There is no env called '{env_name}'"
        return getattr(EnvFactory, env_name.lower())


if False:
    class OsimWrapper(OsimEnv):

        def __init__(self, env, visualize=True, integrator_accuracy=5e-5, report=None):
            self.model_path = env.opensim_model_file
            self.env = env

            # Load model
            super(OsimWrapper, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy)

            # Set timestep
            self.osim_model.stepsize = 0.002

            # initialize state
            state = self.osim_model.model.initializeState()
            self.osim_model.set_state(state)

            # Get joint names
            self.joint_names = self.get_observation_names()

            # Get muscle names, this is the order control values should be in when input to step method
            self.muscle_names = [muscle.getName() for muscle in self.osim_model.muscleSet]
            #self.muscle_mapping = {muscle: self.muscle_names.index(muscle) for muscle in self.muscle_names}

            # Check if we want to save simulated states
            if report:
                bufsize = 0
                self.observations_file = open('%s-obs.csv' % (report,), 'w', bufsize)
                self.actions_file = open('%s-act.csv' % (report,), 'w', bufsize)
                self.get_headers()

        def reset(self, project=True, obs_as_dict=True):

            # initialize state
            state = self.osim_model.model.initializeState()

            # Get joint positions and velocities
            Q = state.getQ()
            QDot = state.getQDot()

            # Set joint positions
            for joint_idx, joint_name in enumerate(self.joint_names):
                Q[joint_idx] = self.env.initial_states["joints"][joint_name]["qpos"]
                QDot[joint_idx] = self.env.initial_states["joints"][joint_name]["qvel"]

            # How to set initial muscle activations? Are they even used since we overwrite them instantly?

            # Set joint positions and velocities
            state.setQ(Q)
            state.setU(QDot)
            self.osim_model.set_state(state)
            self.osim_model.model.equilibrateMuscles(self.osim_model.state)

            # Set time to zero
            self.osim_model.state.setTime(0)
            self.osim_model.istep = 0
            self.t = 0

            self.osim_model.reset_manager()

            return self.get_observation()

        def load_model(self, model_path=None):
            super(OsimWrapper, self).load_model(model_path)

        def step(self, action):

            obs, reward, done, info = super(OsimWrapper, self).step(action, project=True, obs_as_dict=False)
            self.t += self.osim_model.stepsize

            return obs, reward, done, info

        def is_done(self):
            return False

        def get_observation_dict(self):

            # Return only joint positions
            states = self.get_state_desc()
            obs = {}
            for joint_name, mapping in self.env.osim_mapping.items():
                obs[joint_name] = states["joint_pos"][mapping[0]][mapping[1]]

            return obs

        def get_observation(self):
            return np.fromiter(self.get_observation_dict().values(), dtype=float)

        def get_observation_names(self):
            return list(self.get_observation_dict().keys())

        def get_state_desc(self):
            d = super(OsimWrapper, self).get_state_desc()
            return d

        def get_reward(self):
            return 0
