# James model 0. RD Karbowski with variable numbers of guidance
# molecules and TC populations and runtime setting of parameters to
# examine behaviour.
enable_language(CXX)
add_executable(gcal gcal.cpp)
target_compile_definitions(gcal PUBLIC FLT=float COMPILE_PLOTTING)
target_link_libraries(gcal ${MORPH_LDFLAGS} ${HDF5LIBLINK} ${JSONLIBLINK})
