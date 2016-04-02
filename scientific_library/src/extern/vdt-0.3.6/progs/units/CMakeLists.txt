# Enable this directory's flags:
SET(CMAKE_CXX_FLAGS ${DIAG_FLAGS})

include_directories ( ${INC_DIR} )
include_directories ( ${INC_DIR}/diagnostic )

# Utils
add_executable ( fpDifferentBit fpDifferentBit.cpp )
add_executable ( printInstSet printInstSet.cpp )
add_executable ( test_randomPool test_randomPool.cpp  )
add_exe_and_link ( test_fcnResponse test_fcnResponse.cpp  None)
add_exe_and_link ( test_fcnComparison test_fcnComparison.cpp None  )
add_exe_and_link ( test_fcnPerformance test_fcnPerformance.cpp rt )

# Funtions
add_exe_and_link ( exp_test exp_test.cpp None )
add_exe_and_link ( log_test log_test.cpp None )
add_exe_and_link ( sqrt_test sqrt_test.cpp None )
add_exe_and_link ( sin_test sin_test.cpp None )
add_exe_and_link ( cos_test cos_test.cpp None )
add_exe_and_link ( inv_test inv_test.cpp None )
add_exe_and_link ( tan_test tan_test.cpp None )
add_exe_and_link ( asin_test asin_test.cpp None )
add_exe_and_link ( acos_test acos_test.cpp None )
add_exe_and_link ( atan_test atan_test.cpp None )
add_exe_and_link ( atan2_test atan2_test.cpp None )

# add_executable ( deleteme deleteme.cpp )
