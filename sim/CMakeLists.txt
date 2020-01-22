# GCAL MODELS BASED ON STEVENS ET AL. (2013) J. NEUROSCI.
enable_language(CXX)

add_executable(gcal gcal.cpp)
target_compile_definitions(gcal PUBLIC FLT=float COMPILE_PLOTTING)
target_link_libraries(gcal ${MORPH_LDFLAGS} ${HDF5LIBLINK} ${JSONLIBLINK} ${OpenCV_LIBS})
