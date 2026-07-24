if ("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
endif()

# Auto-detech Soc type and Soc version, if detect failed, will abort build
set(SOC_VERSION "")
function(detect_ascend_soc_type SOC_VERSION)
    execute_process(
        COMMAND bash -c "npu-smi info|awk -F' ' 'NF > 0 && NR==7 {print $3}'"
        OUTPUT_VARIABLE npu_info
        RESULT_VARIABLE npu_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if("${npu_info}" STREQUAL "" OR ${npu_result})
        message(FATAL_ERROR "Auto-detech ascend soc type failed, please specify manually or check ascend device working normally.")
    endif()
    set(${SOC_VERSION} "Ascend${npu_info}" PARENT_SCOPE)
endfunction()

if(NOT SOC_TYPE)
    detect_ascend_soc_type(SOC_VERSION)
    set(SOC_TYPE "${SOC_VERSION}")
    message(STATUS "CANN: SOC_VERSION auto-detected is:${SOC_VERSION}")
endif()

string(TOLOWER ${SOC_TYPE} SOC_VERSION) # SOC_VERSION need lower

# Construct Soc specify compile option: ASCEND_#Soc_Major_SN. Such as ASCEND_910B, ASCEND_310P.
string(REGEX MATCH "[0-9]+[a-zA-Z]" SOC_TYPE_MAJOR_SN "${SOC_VERSION}")
set(SOC_TYPE_COMPILE_OPTION "ASCEND_${SOC_TYPE_MAJOR_SN}")
string(TOUPPER ${SOC_TYPE_COMPILE_OPTION} SOC_TYPE_COMPILE_OPTION)
message(STATUS "CANN: SOC_VERSION =  ${SOC_VERSION}")
option(USE_ACL_GRAPH "Enable CANN graph execution (ACL graph mode)" OFF)
option(GGML_CANN_ATB "Enable optional ATB experimental operators" OFF)
set(GGML_CANN_ATB_ROOT "$ENV{ATB_HOME_PATH}" CACHE PATH "ATB cxx_abi_1 installation root")

if(USE_ACL_GRAPH AND (SOC_TYPE_MAJOR_SN STREQUAL "310P" OR SOC_TYPE_COMPILE_OPTION STREQUAL "ASCEND_310P"))
    message(FATAL_ERROR
        "CANN Graph (ACL graph mode) is not supported on 310P devices. "
        "Please build with -DUSE_ACL_GRAPH=OFF or use a supported SOC.")
endif()

if (CANN_INSTALL_DIR)
    # Only Support Linux.
    if (NOT UNIX)
        message(FATAL_ERROR "CANN: CANN toolkit supports unix but not ${CMAKE_SYSTEM_NAME}")
    endif()

    # Supported platforms: x86-64, arm64
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    else()
        message(FATAL_ERROR "CANN: CANN toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # Set header and libs
    set(CANN_INCLUDE_DIRS
        ${CANN_INSTALL_DIR}/include
        ${CANN_INSTALL_DIR}/include/aclnn
        ${CANN_INSTALL_DIR}/acllib/include
    )

    list(APPEND CANN_LIBRARIES
        ascendcl
        nnopbase
        opapi
        acl_op_compiler
    )

    file(GLOB GGML_SOURCES_CANN "*.cpp")

    ggml_add_backend_library(ggml-cann ${GGML_SOURCES_CANN})
    target_link_libraries(ggml-cann PRIVATE ${CANN_LIBRARIES})
    target_include_directories(ggml-cann PRIVATE ${CANN_INCLUDE_DIRS})
    target_link_directories(ggml-cann PRIVATE ${CANN_INSTALL_DIR}/lib64)

    target_compile_definitions(ggml-cann PRIVATE "-D${SOC_TYPE_COMPILE_OPTION}")

    if (GGML_CANN_ATB)
        if (NOT GGML_CANN_ATB_ROOT)
            set(GGML_CANN_ATB_ROOT "/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1")
        endif()
        find_path(GGML_CANN_ATB_INCLUDE_DIR atb/atb_infer.h
            PATHS "${GGML_CANN_ATB_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
        find_library(GGML_CANN_ATB_LIBRARY atb
            PATHS "${GGML_CANN_ATB_ROOT}/lib" "${GGML_CANN_ATB_ROOT}/lib64"
            NO_DEFAULT_PATH REQUIRED)
        find_library(GGML_CANN_ATB_MIXOPS_LIBRARY atb_mixops
            PATHS "${GGML_CANN_ATB_ROOT}/lib" "${GGML_CANN_ATB_ROOT}/lib64"
            NO_DEFAULT_PATH REQUIRED)
        target_sources(ggml-cann PRIVATE atb_ops.cc)
        target_include_directories(ggml-cann PRIVATE "${GGML_CANN_ATB_INCLUDE_DIR}")
        target_compile_definitions(ggml-cann PRIVATE GGML_CANN_USE_ATB=1)
        target_link_libraries(ggml-cann PRIVATE
            "${GGML_CANN_ATB_LIBRARY}" "${GGML_CANN_ATB_MIXOPS_LIBRARY}")
        get_filename_component(GGML_CANN_ATB_LIBRARY_DIR "${GGML_CANN_ATB_LIBRARY}" DIRECTORY)
        set_property(TARGET ggml-cann APPEND PROPERTY BUILD_RPATH "${GGML_CANN_ATB_LIBRARY_DIR}")
        message(STATUS "CANN: experimental ATB operators enabled from ${GGML_CANN_ATB_ROOT}")
    else()
        message(STATUS "CANN: experimental ATB operators disabled")
    endif()

    if (USE_ACL_GRAPH)
        target_compile_definitions(ggml-cann PRIVATE USE_ACL_GRAPH)
        message(STATUS "CANN: USE_ACL_GRAPH is enabled.")
    else()
        message(STATUS "CANN: USE_ACL_GRAPH is disabled.")
    endif()

    message(STATUS "CANN: CANN_INCLUDE_DIRS =  ${CANN_INCLUDE_DIRS}")
    message(STATUS "CANN: CANN_LIBRARIES =  ${CANN_LIBRARIES}")
else()
    message(FATAL_ERROR "CANN: Can't find CANN_INSTALL_DIR, did you forget to source set_var.sh?")
endif()
