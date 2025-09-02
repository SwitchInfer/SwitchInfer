#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <hwloc.h>
#include <sched.h>
#include <pthread.h>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>


#ifdef  __cplusplus
extern "C" {
#endif

void set_layer_param_dir_config(const char * model_name);
char * get_layer_param_dir_c_str();
char * get_layer_param_config_c_str();
void load_layer_param();

#define LAYER_PARAM_PREFIX       "/layer_param"
#define LAYER_PARAM_DIR          (get_layer_param_dir_c_str())
#define LAYER_PARAM_CONFIG       (get_layer_param_config_c_str())

static int find_layer_param_config() {
    struct stat st;
    return stat(LAYER_PARAM_CONFIG, &st) == 0;
}

#define WRITE_LAYER_PARAM   (!find_layer_param_config())

#define EMBED_BATCH_SIZE              16  // 16 on arm
#define GEMM_CHUNK_SIZE_PREFILL       16  // must be multiple of VEC_DOT_LOOP_UNROLL
#define GEMM_CHUNK_SIZE_DECODE        16  // must be multiple of VEC_DOT_LOOP_UNROLL
#define VEC_DOT_LOOP_UNROLL_4         4   // cannot be changed
#define VEC_DOT_LOOP_UNROLL_2         2   // cannot be changed
#define VEC_DOT_LOOP_UNROLL_1         1   // cannot be changed

extern int64_t reference_round;
extern int64_t layer_id;

extern struct ggml_compute_state *workers_;

#define L1_DIST  512
#define L2_DIST  1024
#define L3_DIST  2048

    struct layer_graph {
      struct ggml_tensor * start;
      int start_node_num;
      struct ggml_tensor * end;
      int end_node_num;
    };

#define CPU_PER_CORE_MAX    16
#define CORE_MAX            512

    struct numa_info {
      int core_id;
      int cpu_ids[CPU_PER_CORE_MAX];
    };

#define LAYER_UPPER_BOUND   256
    extern int64_t total_layers_num; // layer num
    extern struct layer_graph layer_graph_[LAYER_UPPER_BOUND]; // layer_graph_[total_layers_num] is the last subgraph for result preparation

    extern int core_num;
    extern int numa_num;
    extern int cpus_per_core;
    extern int thread_per_numa;
    extern struct numa_info core_info[CORE_MAX];

static void set_numa_core_info() {
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    // NUMA 节点信息
    int numa_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
    int num_numa_nodes = hwloc_get_nbobjs_by_depth(topology, numa_depth);
    numa_num = num_numa_nodes;

    // 获取物理核心信息并填充 core_info[]
    core_num = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);

    cpus_per_core = 1; // no SMT

    for (int i = 0; i < core_num; ++i) {
        hwloc_obj_t core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, i);
        hwloc_bitmap_t cpuset = hwloc_bitmap_dup(core->cpuset);
        hwloc_bitmap_singlify(cpuset);  // 取一个 PU 启动线程绑定更安全
        hwloc_obj_t pu = NULL;
        int cpu_count = 0;

        // 遍历 core->cpuset 中的所有 PU（逻辑 CPU）
        for (int cpu = hwloc_bitmap_first(core->cpuset);
             cpu != -1;
             cpu = hwloc_bitmap_next(core->cpuset, cpu)) {

            core_info[i].cpu_ids[cpu_count++] = cpu;
        }

        core_info[i].core_id = i;

    }

    thread_per_numa = core_num / numa_num;

    hwloc_topology_destroy(topology);

}


    struct memblk_info {
        int size;
        void *ptr;
        int *fd; // int [REMAP_THREAD_NUM]
    };

#define MEM_BLOCK_NUM       16
#define NUMA_MAX_NODE       128
    extern uint8_t *            numa_work_data[NUMA_MAX_NODE];
    extern struct memblk_info * numa_mem_info[NUMA_MAX_NODE];
    extern struct memblk_info   numa_l_out_info[LAYER_UPPER_BOUND];
    extern struct memblk_info   k_cache_info[LAYER_UPPER_BOUND];
    extern struct memblk_info   v_cache_info[LAYER_UPPER_BOUND];

    enum LLamaModelLayerNum {
        ln_1B        = 16,
        ln_3B        = 28,
        ln_8B        = 32,
    };

    enum LlamaModelDecodeChunk {
        dc_1B        = 64,
        dc_3B        = 144,
        dc_8B        = 224,
    };

    extern int decode_chunk_size_;

#define PAGE_SIZE           4096

#define INTERLEAVE_GAP      (decode_chunk_size_) // unit: pages, 16 on arm server, 64 on intel server
#define LAYER_PARAM_NUM     7
    extern struct memblk_info   layer_param_info[LAYER_UPPER_BOUND][LAYER_PARAM_NUM];

    enum InferStage {
        PREFILL       = 0,
        SWITCHING_P2D = 1,
        SWITCHING_D2P = 2,
        DECODE        = 3,
    };

    extern enum InferStage      cur_stage;

    enum SwitchStage {
        PREFILL_2_DECODE = 0,
        DECODE_2_PREFILL = 1,
        NONE_SWITCH      = 2,
    };

    extern enum SwitchStage     cur_switch_stage;
    extern char                 layer_infer_done[LAYER_UPPER_BOUND];
    extern int64_t              cur_switch_start_layer;
    extern int64_t              cur_switch_remain_layer;

#define MAX_REMAP_THREAD_NUM    1024
#define MAX_REMAP_BATCH_NUM     16
#define MIN_REMAP_BATHC_LAYER   2
#define REMAP_THREAD_NUM        32
#define OPEN_FILE_LIMIT         65536
    extern int       remap_thread_num; // deprecated
    extern pthread_t remap_threads[MAX_REMAP_THREAD_NUM];
    void ggml_barrier_remap();

#define KV_CACHE_LIMIT          1024
#define BATCH_SIZE_LIMIT        256
#define SLIDE_WINDOW_NUM        4
#define SLIDE_WINDOW_SIZE       10

    struct chunk_queue {
        int cur_chunk __attribute__((aligned(64)));
        int max_chunk __attribute__((aligned(64)));
        pthread_mutex_t lock __attribute__((aligned(64)));
    };

    extern struct chunk_queue gemm_chunk_queue[NUMA_MAX_NODE] __attribute__((aligned(64)));

    static void init_gemm_queue() {
        for (int n = 0; n < NUMA_MAX_NODE; n++) {
            gemm_chunk_queue[n].cur_chunk = 0;
            gemm_chunk_queue[n].max_chunk = 0;
            pthread_mutex_init(&gemm_chunk_queue[n].lock, NULL);
        }
    }

    static void set_gemm_queue_numa(const int node, const int cur_chunk, const int max_node) {
        gemm_chunk_queue[node].cur_chunk = cur_chunk;
        gemm_chunk_queue[node].max_chunk = max_node;
    }

    static int get_next_chunk(const int node) {
        int chunk;
        pthread_mutex_lock(&gemm_chunk_queue[node].lock);
        chunk = gemm_chunk_queue[node].cur_chunk;
        if (chunk < gemm_chunk_queue[node].max_chunk) {
            gemm_chunk_queue[node].cur_chunk++;
        }
        pthread_mutex_unlock(&gemm_chunk_queue[node].lock);
        return chunk >= gemm_chunk_queue[node].max_chunk ? -1 : chunk;
    }

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

#ifdef __cplusplus
}
#endif
