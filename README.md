# OpenSim to MuJoCo XML converter

| :exclamation:  This project is deprecated, but the development continues in [MyoConverter](https://github.com/MyoHub/myoconverter). |
|-------------------------------------------------------------------------------------------------------------------------------------|


If you use these converted MuJoCo models in your research, please cite ["Converting Biomechanical Models from OpenSim to MuJoCo"](https://arxiv.org/abs/2006.10618), as well as the original models. The converted models, and the original ones, are available for academic and other non-commercial use (check the references given in table below for licenses of original models).

Disclaimer: I started writing this code when I was still learning MuJoCo and OpenSim, and therefore the code logic is subpar at some places and could use a little update.


## Models

| Converted file | Original file | File containing optimized parameters | Description of the model | Reference |
|---|---|---|---|---|
| [gait10dof18musc MuJoCo](https://github.com/aikkala/O2MConverter/tree/master/models/converted/gait10dof18musc_converted) | [gait10dof18musc OpenSim](https://github.com/aikkala/O2MConverter/blob/master/models/opensim/Gait10dof18musc/gait10dof18musc.osim) | [data.pckl](https://github.com/aikkala/O2MConverter/blob/master/tests/gait10dof18musc/output/data.pckl) | A simple leg model consisting of both legs and rotating torso. Derived from the *gait2392* model below. | Distributed with OpenSim,  [OpenSim web page (download requires registration)](https://simtk.org/frs/download.php?file_id=4081) [GitHub](https://github.com/opensim-org/opensim-models/tree/master/Models/Gait10dof18musc)  |
| [MoBL_ARMS MuJoCo](https://github.com/aikkala/O2MConverter/tree/master/models/converted/MoBL_ARMS_model_for_mujoco_converted) | [MoBL_ARMS OpenSim](https://github.com/aikkala/O2MConverter/blob/master/models/opensim/MoBL_ARMS_OpenSim_tutorial_33/MoBL_ARMS_model_for_mujoco.osim) | [data.pckl](https://github.com/aikkala/O2MConverter/blob/master/tests/mobl_arms/output/data.pckl) | A dynamic shoulder and arm model with fixed torso. | [Project web page (download requires registration)](https://simtk.org/frs/?group_id=657). [*Benchmarking of Dynamic Simulation Predictions in Two Software Platforms Using an Upper Limb Musculoskeletal Model*, K. R. Saul, X. Hu, C. M. Goehler, M. E. Vidt, M. Daly, A. Velisar, W. M. Murray (2014)](https://pubmed.ncbi.nlm.nih.gov/24995410/) |
| [gait2392 MuJoCo](https://github.com/aikkala/O2MConverter/tree/master/models/converted/MoBL_ARMS_model_for_mujoco_converted) | [gait2392 OpenSim](https://github.com/aikkala/O2MConverter/blob/master/models/opensim/Gait2392_Simbody/gait2392_millard2012muscle.osim) | [data.pckl](https://github.com/aikkala/O2MConverter/blob/master/tests/gait2392/output/data.pckl) | A leg model consisting of both legs and a rotating/bending torso. | Distributed with OpenSim, [OpenSim web page (download requires registration)](https://simtk.org/frs/download.php?file_id=4081) [GitHub](https://github.com/opensim-org/opensim-models/tree/master/Models/Gait2392_Simbody) |
| [leg6dof9musc MuJoCo](https://github.com/aikkala/O2MConverter/tree/master/models/converted/leg6dof9musc_converted) | [leg6dof9musc OpenSim](https://github.com/aikkala/O2MConverter/blob/master/models/opensim/Leg6Dof9Musc/leg6dof9musc.osim) | - | A simple leg model (one leg), derived from *gait2392* model. | Distributed with OpenSim, [OpenSim web page (download requires registration)](https://simtk.org/frs/download.php?file_id=4081) [GitHub](https://github.com/opensim-org/opensim-models/tree/master/Models/Leg6Dof9Musc) |
| [Hamner2010](https://github.com/aikkala/O2MConverter/tree/master/models/converted/FullBodyModel_SimpleArms_Hamner2010_Markers_v2_0_converted) | Not available in this repo, see reference | - | 3D full-body model with 92 musculotendon actuators in lower extremities | [Project web page (download requires registration)](https://simtk.org/projects/runningsim). [*Muscle contributions to propulsion and support during running*, S.R. Hamner, A. Seth, S.L. Delp (2010)](https://pubmed.ncbi.nlm.nih.gov/20691972/)|
| [Rajagopal2015](https://github.com/aikkala/O2MConverter/tree/master/models/converted/Rajagopal2015_converted) | Not available in this repo, see reference | - | 3D full-body model with a muscle-actuated lower extremity and torque actuated trunk and upper extremity | [Project web page (download requires registration)](https://simtk.org/projects/full_body). [*Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait*, A. Rajagopal, C. L. Dembia, M. S. DeMers, D. D. Delp, J. L. Hicks, S. L. Delp (2016)](https://pubmed.ncbi.nlm.nih.gov/20691972/)|
|[HYOID1.2](https://github.com/aikkala/O2MConverter/tree/master/models/converted/HYOID_1.2_ScaledStrenght_UpdatedInertia_converted) | [HYOID1.2 OpenSim](https://github.com/aikkala/O2MConverter/tree/master/models/opensim/HYOID/) | - | Neck model | [Project web page (download requires registration)](https://simtk.org/projects/neckdynamics#). [*The Inclusion of Hyoid Muscles Improve Moment Generating Capacity and Dynamic Simulations in Musculoskeletal Models of the Head and Neck*, J. D. Mortensen, A. N. Vasavada, A. S. Merryweather (2018)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199912) |


## Limitations and known bugs

- Doesn't work with OpenSim 4.0 (or later) models
- PathPoint names must have the same prefix and a running index, which defines the order of tendon path. For instance, in *MoBL_ARMS* model musculo-tendon unit *DELT2* has PathPoints *{"DELT2-P1", "default", "DELT2-P3", "DELT2-P4"}*, and thus the PathPoint named _"default"_ should be changed to _"DELT2-P2"_. This problem occurs because the XML parser we're using scrambles the order of individual PathPoints in certain situations. There is a pull request to fix this behaviour, but it hasn't been merged yet (and the pull request is from 2016 so chances are it won't be merged).
- For some of the converted models the convex hulls around meshes/geometries are too close to each other and they collide. A possible remedy is to disable contacts in the converted model (although then there's a risk of physically impossible trajectories).
- The order in which *TransformAxis* are processed for custom joints is likely to be incorrect, although it works for the models presented here. If a converted model doesn't look like the original OpenSim model, then this might be the problem.
- *PointActuator* and *TorqueActuator* are ignored; *WrappingObjects* are ignored; there probably are other other OpenSim model specifications that I haven't encountered yet and are ignored as well
- The *worldbody* is assumed to be called *ground* in an OpenSim model

## How to run the converter

Easiest way to run the converter is to use conda to create the environment from *conda_env.yml*, activate the environment, and then run *O2MConverter.py* script

```
> conda env create --name O2MConverter --file=conda_env.yml
> conda activate O2MConverter
> python O2MConverter.py opensim_model_file.osim /location/where/converted/model/will/be/saved /location/where/geometry/files/are
```

## How to optimize converted model's parameters (for a new model)

- **NOTE (added 2022 July)** This parameter optimization approach may be partially outdated (e.g. some function input parameters may have changed etc). Use at your own risk. Also note that this approach is intended to be used only on models that are fixed in air (like the MoBL ARMS model)

- Use conda to create the environment from *conda_env_for_testing.yml*. Note that this installs MuJoCo, and you must have set the environment variable *MUJOCO_PY_MJKEY_PATH* (pointing to your MuJoCo license file) prior to creating the environment.

- Convert the OpenSim model as instructed above, but this time input *true* as fourth argument to O2MConverter.py. This disables contacts in the MuJoCo model since they are disabled in OpenSim forward simulations also.

- Create a new template in *tests/envs.py* using the *EnvTemplate* class. Note that you need an OpenSim forward dynamics setup XML file (you can just copy and modify an existing one from *models/opensim/[any_model]*) and also an initial states file (this one's more tricky to create, easiest just to run forward dynamics simulation in OpenSim and use the output file with states). Also modify the function *get(model_name)* in *tests/envs.py* to return this new env.

- Run script *tests/generate_controls.py* using the model name you specified in the previous step for the *get* function as input argument. This creates a hundred muscle excitation sets in a folder specified by the *EnvTemplate* object.

- Run script *tests/run_opensim_simulations.py* (using the model name as input argument) to run the OpenSim simulations with the generated muscle excitation sets

- Run script *tests/optimize_mujoco_parameters.py* (using the model name as input argument) to optimize the converted model parameters. The parameters will be saved in a file specified by the *EnvTemplate* object.

## How to load converted model's optimized parameters

The optimized parameters are saved in *tests/[model_name]/output/data.pckl*, or in a file specified by the *EnvTemplate* object if you created & optimized a new model. Use function *load_data(args)* from *Utils.py* to load the optimized parameters, and function *set_parameters(args)* to set the models into a *mujoco_py.PyMjModel* model (see function *run_mujoco_simulations(args)* in *tests/process_test_runs.py* for an example). 

## Contributors

[Florian Fischer](https://github.com/fl0fischer), [Miroslav Bachinski](https://bachinski.de/) (analysis and conversion of musculotendon properties)