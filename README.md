# SelfOrganisingMaps

Implementation of GCAL model reported in Stevens et al., (2013) J. Neurosci. paper.

First install morphologica (https://github.com/ABRG-Models/morphologica), then build in the usual cmake way

mkdir build
cd build
cmake .
make
cd ..


Then run model using e.g., 

./build/sim/gcal configs/config.js 1 1 2

The final 3 numbers are
1. random seed
2. Mode -- 0: displays-off, 1: displays-on, 2: map-only
3. Training pattern type -- 0: oriented Gaussian, 1: Preloaded vectors from hdf5 file specificed in config, 2: videocamera input

If the path to file of a saved weightfile is optionally appended, these weights will be used, else initial weights are random.

Enjoy!
