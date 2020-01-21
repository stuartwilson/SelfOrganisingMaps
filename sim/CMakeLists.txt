# GCAL MODELS BASED ON STEVENS ET AL. (2013) J. NEUROSCI.
enable_language(CXX)

add_executable(model model.cpp)
target_compile_definitions(model PUBLIC FLT=float COMPILE_PLOTTING)
target_link_libraries(model ${MORPH_LDFLAGS} ${HDF5LIBLINK} ${JSONLIBLINK})

add_executable(model_ExtIn model_ExtIn.cpp)
target_compile_definitions(model_ExtIn PUBLIC FLT=float COMPILE_PLOTTING)
target_link_libraries(model_ExtIn ${MORPH_LDFLAGS} ${HDF5LIBLINK} ${JSONLIBLINK})
