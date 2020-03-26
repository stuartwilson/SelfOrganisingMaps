# SelfOrganisingMaps

Implementation of GCAL model reported in Stevens et al., (2013) J. Neurosci. paper.

First install morphologica (https://github.com/ABRG-Models/morphologica), then build in the usual cmake way

mkdir build &nbsp;
cd build &nbsp;
cmake .. &nbsp;
make &nbsp;
cd .. &nbsp;


Then run model using e.g., 

./build/sim/gcal configs/config.json 1 1 2

The final 3 numbers are
1. random seed
2. Mode -- 0: displays-off, 1: displays-on, 2: map-only
3. Training pattern type -- 0: oriented Gaussians, 1: preloaded vectors*, 2: videocamera input

*Note that if using preloaded vectors you will need to supply a hdf5 file as the "patterns" parameter in the json file. An example can be generated by running 'python genTestPatterns.py' from the configs folder to generate 'configs/testPatterns.h5'

If the path to file of a saved weightfile is optionally appended, these weights will be used, else initial weights are random.

Enjoy!
