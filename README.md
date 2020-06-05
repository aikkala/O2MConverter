# OpenSim to MuJoCo XML converter (work in progress).

If you use these converted MuJoCo models in your research, please cite **TBA**, as well as the original models.


## Models

| Model Name      | Short Summary                                                                                           | Reference                                                                                                                                                                                                                                                             |
|-----------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gait10dof18musc | A simple leg model consisting of both legs and rotating torso. Derived from the *gait2392* model below. | Original model name and file name: **Gait10dof18musc**, **gait10dof18musc.osim**. Distributed with OpenSim,  [OpenSim web page (download requires registration)](https://simtk.org/frs/download.php?file_id=4081) [GitHub](https://github.com/opensim-org/opensim-models/tree/master/Models/Gait10dof18musc)  |
| mobl_arms       | A dynamic shoulder and arm model with fixed torso.                                                      | Original model file: **MoBL_ARMS_module6_7_CMC.osim**. [Project web page (download requires registration)](https://simtk.org/frs/?group_id=657). [*Benchmarking of Dynamic Simulation Predictions in Two Software Platforms Using an Upper Limb Musculoskeletal Model*, K. R. Saul, X. Hu, C. M. Goehler, M. E. Vidt, M. Daly, A. Velisar, W. M. Murray](https://pubmed.ncbi.nlm.nih.gov/24995410/) |
| gait2392        | A leg model consisting of both legs and a rotating/bending torso.                                       | Original model name and file name: **Gait2392_Simbody**, **gait2392_millard2012muscle.osim**. Distributed with OpenSim, [OpenSim web page (download requires registration)](https://simtk.org/frs/download.php?file_id=4081) [GitHub](https://github.com/opensim-org/opensim-models/tree/master/Models/Gait2392_Simbody) |


## Limitations and known bugs

- Doesn't work with OpenSim 4.0 (or later) models
- PathPoint names must have the same prefix and a running index, which defines the order of tendon path. For instance, in *MoBL_ARMS* model musculo-tendon unit *DELT2* has PathPoints *{"DELT2-P1", "default", "DELT2-P3", "DELT2-P4"}*, and thus the PathPoint named _"default"_ should be changed to _"DELT2-P2"_. This problem occurs because the XML parser we're using scrambles the order of individual PathPoints in certain situations. There is a pull request to fix this behaviour, but it hasn't been merged yet (and the pull request is from 2016 so chances are it won't be merged).





## How to run the converter

Easiest way to run the converter is to use conda to create the environment from *O2MConverter\_without\_mujoco.yml* (TBD), activate the environment, and then run *O2MConverter.py* script

```
> conda env create --name O2MConverter --file=O2MConverter_without_mujoco.yml
> conda activate O2MConverter`
> python O2MConverter.py opensim\_model\_file.osim /location/where/converted/model/will/be/saved /location/where/geometry/files/are
```

