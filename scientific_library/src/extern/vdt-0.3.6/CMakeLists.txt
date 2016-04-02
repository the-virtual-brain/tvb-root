# VDT Math LIbrary
cmake_minimum_required (VERSION 2.6)
project (Vdt)

#-------------------------------------------------------------------------------
# Include the defaults
include ( CMakeDefaults.txt )
#-------------------------------------------------------------------------------

# configuration options -- you may change them when running cmake ==============
# with 'cmake -D <OPT>=<value> .'

option( DIAG "Build in diagnostic mode - all diagnostic exes (default cache entry: OFF)" OFF)
option( AVX "Use AVX instruction set (default cache entry: OFF)" OFF)
option( AVX2 "Use AVX2 instruction set (default cache entry: OFF)" OFF)
option( SSE "Use SSE instruction set (default cache entry: ON)" ON)
option( NEON "Use NEON instruction set (default cache entry: OFF)" OFF)
option( BUILD_SHARED_LIBS "Build libraries as SHARED instead of STATIC (default cache entry: ON)" ON)
option( PRELOAD "Create in the library the symbols to preload the library (default cache entry: OFF)" OFF)
option( USE_VC "Use Vc library - requires symlink to Vc from ${CMAKE_SOURCE_DIR} (default cache entry: OFF)" OFF)
option( DEBUG "Compile library with debug symbols (default is OFF)" OFF)

# determine compiler type and version ==========================================
set(GCC_THRESHOLD_VERSION "4.7.0")
set(CLANG_THRESHOLD_VERSION "3.3")
set(ICC_THRESHOLD_VERSION "13.0")

set(CMAKE_COMPILER_IS_NEW_GCC "FALSE")
set(CMAKE_COMPILER_IS_OLD_GCC "FALSE")
set(CMAKE_COMPILER_IS_ICC "FALSE")
set(CMAKE_COMPILER_IS_OLD_ICC "FALSE")
set(CMAKE_COMPILER_IS_NEW_ICC "FALSE")
set(CMAKE_COMPILER_IS_CLANG "FALSE")
set(CMAKE_COMPILER_IS_NEW_CLANG "FALSE")
set(CMAKE_COMPILER_IS_OLD_CLANG "FALSE")
set(CMAKE_COMPILER_IS_UNSUPPORTED "FALSE")

set(COMPILER_VERSION "0.0")


getCompilerVersion()
  
if(CMAKE_CXX_COMPILER MATCHES "clang")
  set(CMAKE_COMPILER_IS_CLANG "TRUE")
  if("${COMPILER_VERSION}" VERSION_LESS ${CLANG_THRESHOLD_VERSION})
    set(CMAKE_COMPILER_IS_OLD_CLANG "TRUE")
    message(STATUS "Found CLANG older than ${CLANG_THRESHOLD_VERSION}")
  else()
    set(CMAKE_COMPILER_IS_NEW_CLANG "TRUE")
    message(STATUS "Found CLANG greater than ${CLANG_THRESHOLD_VERSION}")
  endif()  

elseif(CMAKE_CXX_COMPILER MATCHES "/(icpc|icc)$")
  set(CMAKE_COMPILER_IS_ICC "TRUE")
  if("${COMPILER_VERSION}" VERSION_LESS ${ICC_THRESHOLD_VERSION})
    set(CMAKE_COMPILER_IS_OLD_ICC "TRUE")
    message(STATUS "Found ICC older than ${ICC_THRESHOLD_VERSION}")
  else()
    set(CMAKE_COMPILER_IS_NEW_ICC "TRUE")
    message(STATUS "Found ICC greater than ${ICC_THRESHOLD_VERSION}")
  endif()  
  
elseif(CMAKE_COMPILER_IS_GNUCXX)
    if("${COMPILER_VERSION}" VERSION_LESS ${GCC_THRESHOLD_VERSION})
    set(CMAKE_COMPILER_IS_OLD_GCC "TRUE")
    message(STATUS "Found GCC older than ${GCC_THRESHOLD_VERSION}")
  else()
    set(CMAKE_COMPILER_IS_NEW_GCC "TRUE")
    message(STATUS "Found GCC greater than ${GCC_THRESHOLD_VERSION}")
  endif()
endif()

if (NOT(CMAKE_COMPILER_IS_GNUCXX OR
        CMAKE_COMPILER_IS_CLANG OR
        CMAKE_COMPILER_IS_ICC))
  set( CMAKE_COMPILER_IS_UNSUPPORTED "TRUE")
  message(STATUS "Unsupported compiler !!!")
endif()

if (CMAKE_COMPILER_IS_OLD_GCC OR
    CMAKE_COMPILER_IS_OLD_CLANG OR
    CMAKE_COMPILER_IS_OLD_ICC OR
    CMAKE_COMPILER_IS_UNSUPPORTED )
  message(STATUS "You are running an old or unsupported compiler: turning off Vc, AVX and DIAG options")
  change_option(USE_VC 0)
  change_option(AVX 0)
  change_option(AVX2 0)
  change_option(DIAG 0)
endif()


# process options that can be processed here ===================================

# SIMD instructions set---------------------------------------------------------

if (NEON)
  message(STATUS "Using NEON instructions!")
  set(SSE_AVX "-mfpu=neon ")
else()
  if (SSE AND (NOT (AVX OR AVX2) ))
    message(STATUS "Using SSE instructions!")
    set(SSE_AVX "-msse")
  endif ()

  if (AVX AND (NOT AVX2))
    message(STATUS "Using AVX instructions!")
    set (SSE_AVX "-mavx")
    if(CMAKE_COMPILER_IS_ICC)
      set(SSE_AVX "-xavx")
    endif()
  endif ()

  if (AVX2)
    message(STATUS "Using AVX2 instructions!")
    set (SSE_AVX "-mavx2")
    if(CMAKE_COMPILER_IS_ICC)
      set(SSE_AVX "-xavx2")
    endif()
  endif ()
endif()

# To use svml at CERN ----------------------------------------------------------
set (INTEL_SVML_FLAGS "")
if (SVML)
  message (STATUS "Linking SVML library")
  set (INTEL_SVML_FLAGS "-mveclibabi=svml -L/afs/cern.ch/sw/IntelSoftware/linux/x86_64/Compiler/11.1/072/lib/intel64/ -lsvml -lirc")
endif (SVML)

# Vc setup ---------------------------------------------------------------------

if(USE_VC)
  message(STATUS "VC usage is turned on now, if you do not intend to use it, run 'cmake -D USE_VC=0 .'")
  set (VC_SYMLINK_MSG "To use Vc you must have a (symlink) 'Vc' leading to the Vc rootdir in your ${CMAKE_SOURCE_DIR}")
  #check for files
  set (VC_LIB_NAME "${CMAKE_SOURCE_DIR}/Vc/libVc.a")
  set (VC_HEADER_NAME "${CMAKE_SOURCE_DIR}/Vc/include/Vc/Vc")

  if(NOT EXISTS ${VC_LIB_NAME})
    message(STATUS "Vc lib not found at ${VC_LIB_NAME}, turning off Vc usage")
    message(STATUS ${VC_SYMLINK_MSG})
    change_option(USE_VC 0)
  endif(NOT EXISTS ${VC_LIB_NAME})

  if (EXISTS ${VC_LIB_NAME})
    if(NOT EXISTS ${VC_HEADER_NAME})
      message(STATUS "Vc header not found at ${VC_HEADER_NAME}, turning off Vc usage")
      message(STATUS ${VC_SYMLINK_MSG})
      change_option(USE_VC 0)
    endif(NOT EXISTS ${VC_HEADER_NAME})
  endif(EXISTS ${VC_LIB_NAME})


  link_directories( ${CMAKE_SOURCE_DIR}/Vc )
endif(USE_VC)

# set compiler options =========================================================

set (LIBTIMING "")
set (LIBTIMINGAPPLE "")
if(DIAG)
  # Library for time measurement: OSx and Linux
  set (LIBTIMING "rt")
  # do not set it if on OSx
  if (APPLE)
    set (LIBTIMINGAPPLE "-framework Carbon")
  endif (APPLE)
endif(DIAG)

#-------------------------------------------------------------------------------
# Compiler optimisations

set (VECT_OPT "-O3 -ffast-math ")
if (CMAKE_COMPILER_IS_ICC)
  set (VECT_OPT "")
endif()

if (CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_COMPILER_IS_CLANG) # for old cmake versions
   set (VECTORIZER_VERBOSITY "-ftree-vectorizer-verbose=0")
   set (INLINE_OPT " --param vect-max-version-for-alias-checks=50 --param inline-unit-growth=150")   
endif()   
if (CMAKE_COMPILER_IS_OLD_GCC OR CMAKE_COMPILER_IS_OLD_CLANG)
 set (CPP11_OPT "-std=gnu++0x")
elseif(CMAKE_COMPILER_IS_ICC)
 set (CPP11_OPT "-std=c++0x")
else()
 set (CPP11_OPT "-std=c++11")
endif()
set (VERBOSITY_OPT "-Winline ")

# set it for clang until it understands __always_inline
if (CMAKE_COMPILER_IS_CLANG)
 set (CLANG_INLINE_DEFINE "-D__extern_always_inline=inline")
else()
 set (CLANG_INLINE_DEFINE "")
endif()

# compiler dependent changes ---------------------------------------------------
if(CMAKE_COMPILER_IS_ICC)
  set (VECTORIZER_VERBOSITY "")
  set (INLINE_OPT "")
elseif(CMAKE_COMPILER_IS_OLD_GCC)
  set (VECT_OPT "-O2")
  set (VECTORIZER_VERBOSITY "")
  set (INLINE_OPT "")
  set (VERBOSITY_OPT "")
  set (CPP11_OPT "")
endif()

set (WARNING_FLAGS "-W -Wall -Werror -Wno-error=unused-parameter")

set (DEBUG_FLAGS " ")
if (DEBUG)
  set (DEBUG_FLAGS " -g")
  message(STATUS "Adding debugging symbols")
endif ()


set (COMMON_FLAGS "${INTEL_SVML_FLAGS} ${SSE_AVX}  ${INLINE_OPT} ${WARNING_FLAGS} ${DEBUG_FLAGS} ${CLANG_INLINE_DEFINE}")
set (LIB_FLAGS "${VERBOSITY_OPT} ${VECT_OPT} ${VECTORIZER_VERBOSITY} ${COMMON_FLAGS}")
set (DIAG_FLAGS "${CPP11_OPT} ${COMMON_FLAGS} ${LIBTIMINGAPPLE} ${VECT_OPT}")

# Locations ====================================================================
# Location of executables
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

# Location of sources
set( SRC_DIR ${CMAKE_SOURCE_DIR}/src )

# Location of library
set( LIB_DIR ${CMAKE_SOURCE_DIR}/lib )

# Common Includes
set (INC_DIR ${CMAKE_SOURCE_DIR}/include )


#-------------------------------------------------------------------------------

add_subdirectory( src )
add_subdirectory( lib )
if (DIAG)
    message("DIAG option is now on, building diagnostic programs")
    add_subdirectory( progs )
    add_subdirectory( progs/units )
else(DIAG)
  message("DIAG option is now off, building library only")
endif(DIAG)

#-------------------------------------------------------------------------------
# Installation

# Install location
INSTALL(FILES
        include/asin.h
        include/atan.h
        include/atan2.h
        include/cos.h
        include/exp.h
        include/identity.h
        include/inv.h
        include/log.h
        include/sincos.h
        include/sin.h
        include/sqrt.h
        include/tan.h
        include/vdtcore_common.h
        include/vdtMath.h
        DESTINATION include/vdt)

