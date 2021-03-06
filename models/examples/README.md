# SelfOrganisingMaps

Implementation of GCAL model reported in Stevens et al., (2013) J. Neurosci. paper. (https://www.jneurosci.org/content/33/40/15747)

First clone morphologica into the SelfOrganisingMaps/models/examples directory (https://github.com/ABRG-Models/morphologica), then build in the usual cmake way

cd SelfOrganisingMaps/models/examples
mkdir build
cd build
cmake ..
make
cd ..

Then run model using e.g., 

```
./build/stevens configs/gcal.json logdir 1 1 1
```

The final 3 numbers are
1. random seed
2. Mode -- 0: displays-off, 1: displays-on
3. Training pattern type -- 0: oriented Gaussians, 1: preloaded vectors*, 2: videocamera input

Run it for a while, 5000 steps or so, and then run ```plotPinwheelDensity.py logdir/measures.h5``` to see how the pinwheel density has been changing over time. It should converge towards pi=3.14

*Note that if using preloaded vectors you will need to supply a hdf5 file as the "patterns" parameter in the json file. An example can be generated by running ```python genTestPatterns.py shouval png ../natural.h5``` from the configs/images folder to generate 'configs/natural.h5'

If the path to file of a saved weightfile is optionally appended to the launch command, then these weights will be used, else initial weights are random.

Enjoy!
