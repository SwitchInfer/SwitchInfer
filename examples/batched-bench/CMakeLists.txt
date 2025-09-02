set(TARGET llama-batched-bench)
add_executable(${TARGET} batched-bench.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

# 添加 libnuma
find_library(NUMA_LIB numa REQUIRED)

target_link_libraries(${TARGET}
        PRIVATE
        common
        llama
        ${CMAKE_THREAD_LIBS_INIT}
        ${NUMA_LIB}              # ✅ 加上这一行
)

# --- hwloc 支持 ----------------------------------------
find_path(HWLOC_INCLUDE_DIR hwloc.h
        HINTS ENV CPATH
        PATH_SUFFIXES include
)

find_library(HWLOC_LIBRARY hwloc
        HINTS ENV LIBRARY_PATH
        PATH_SUFFIXES lib
)

if (HWLOC_INCLUDE_DIR AND HWLOC_LIBRARY)
    message(STATUS "Found hwloc: ${HWLOC_INCLUDE_DIR}, ${HWLOC_LIBRARY}")
    target_include_directories(${TARGET} PRIVATE ${HWLOC_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE ${HWLOC_LIBRARY})
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_HWLOC)
else()
    message(WARNING "hwloc not found - please ensure it is installed and accessible to CMake")
endif()
# --- hwloc 支持 ----------------------------------------

target_compile_features(${TARGET} PRIVATE cxx_std_17)
