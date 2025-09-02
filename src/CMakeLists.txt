llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-grammar.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-kv-cache.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

target_include_directories(llama PUBLIC . ../include ../common)
target_compile_features   (llama PUBLIC cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

# --- NUMA 支持 -----------------------------------------
find_library(NUMA_LIBRARY numa)
if (NUMA_LIBRARY)
    message(STATUS "Found libnuma: ${NUMA_LIBRARY}")
    target_link_libraries(llama PRIVATE ${NUMA_LIBRARY})
    target_compile_definitions(llama PRIVATE GGML_USE_NUMA)
else()
    message(WARNING "libnuma not found - NUMA support will be disabled")
endif()

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
    target_include_directories(llama PRIVATE ${HWLOC_INCLUDE_DIR})
    target_link_libraries(llama PRIVATE ${HWLOC_LIBRARY})
    target_compile_definitions(llama PRIVATE GGML_USE_HWLOC)
else()
    message(WARNING "hwloc not found - please ensure it is installed and accessible to CMake")
endif()
# --- hwloc 支持 ----------------------------------------


if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
