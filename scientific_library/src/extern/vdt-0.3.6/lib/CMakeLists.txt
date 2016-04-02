# Rules for making the so library with the vector and libm-like signatures

# Define this directory's flags:
SET(CMAKE_CXX_FLAGS  ${LIB_FLAGS})

include_directories ( ${INC_DIR} )

if(BUILD_SHARED_LIBS)
  message(STATUS "Libraries are configured as: SHARED")
else()
  message(STATUS "Libraries are configured as: STATIC")
endif()

if(PRELOAD)
  message(STATUS "Symbols for the preload requested")
endif()

# The library
ADD_LIBRARY(vdt ${SRC_DIR}/vdtMath_signatures.cc ${INC_DIR}/vdtMath.h )

# Installation of the lib
INSTALL(TARGETS vdt  
        DESTINATION lib)
        

# Build Vc wrapper (without c++11)
if(USE_VC)
    SET(CMAKE_CXX_FLAGS  "${LIB_FLAGS} -fabi-version=6")
    include_directories( ${CMAKE_SOURCE_DIR}/Vc  ${CMAKE_SOURCE_DIR}/Vc/include )
    add_library(VcWrapper ${SRC_DIR}/vdtdiag_vcWrapper.cc ${SRC_DIR}/vdtdiag_vcWrapper.h)
    target_link_libraries(VcWrapper libVc.a)
endif()
