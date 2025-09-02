#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-aarch64.h"
#include "ggml-cpu-traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <cctype>
#include <cmath>
#include <string>
#include <vector>
#include "numa.h"
#include <sys/mman.h>
#include <numaif.h>
#include <fstream>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/resource.h>

#ifdef GGML_USE_CPU_HBM
#include "ggml-cpu-hbm.h"
#endif

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t>& ggml_backend_cpu_get_extra_buffers_type() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_AARCH64
        if (ggml_backend_cpu_aarch64_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_aarch64_buffer_type());
        }
#endif

        bufts.push_back(NULL);

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    return ggml_backend_cpu_get_extra_buffers_type().data();

    GGML_UNUSED(device);
}

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto extra : ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra && extra == buft) return true;
    }
    return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

// TODO: 单个layer中一轮张量并行所要处理的embedding数量
//#define EMBED_BATCH_SIZE      2    // TODO: numa node core num * embed cache block param, need to be adjusted according to system config
extern enum InferStage cur_stage = InferStage::PREFILL; // PREFILL when reference_num_ = 0, SWITCHING when reference_num_ = 1, DECODE when reference_num_ > 1
int reference_num_ = 0;
int64_t total_layers_num;
struct layer_graph layer_graph_[LAYER_UPPER_BOUND];

struct memblk_info   layer_param_info[LAYER_UPPER_BOUND][LAYER_PARAM_NUM];
struct memblk_info * numa_mem_info[NUMA_MAX_NODE];
struct memblk_info   numa_l_out_info[LAYER_UPPER_BOUND];
struct memblk_info   k_cache_info[LAYER_UPPER_BOUND];
struct memblk_info   v_cache_info[LAYER_UPPER_BOUND];

inline void set_memblk_info(struct memblk_info *mi, struct ggml_tensor *node, int numa_node) {
    mi->size = node->nb[3];
    mi->ptr = numa_alloc_onnode(mi->size, numa_node);
//    memset(mi->ptr, 0, mi->size);
}

void init_mem_block(struct ggml_cgraph * cgraph) {
    struct ggml_tensor **nodes_ = cgraph->nodes;
    for (int n = 0; n < numa_num; n++) {
        numa_mem_info[n] = static_cast<memblk_info *>(malloc(MEM_BLOCK_NUM * sizeof(struct memblk_info)));

        set_memblk_info(&numa_mem_info[n][0], nodes_[0], n);
        set_memblk_info(&numa_mem_info[n][1], nodes_[1], n);
        set_memblk_info(&numa_mem_info[n][2], nodes_[28], n);
        set_memblk_info(&numa_mem_info[n][3], nodes_[5], n);
        set_memblk_info(&numa_mem_info[n][4], nodes_[18], n);
        set_memblk_info(&numa_mem_info[n][5], nodes_[29], n);
    }

    for (int l = 0; l < total_layers_num; l++) {
        set_memblk_info(&numa_l_out_info[l], nodes_[32], (l+1) % numa_num); // input of next layer
        set_memblk_info(&k_cache_info[l], nodes_[10]->src[0], l % numa_num); // k cache
        set_memblk_info(&v_cache_info[l], nodes_[9], l % numa_num); // v cache
    }
}

static int compute_embed_mbatch_size(const int seqlen) {
    // if (seqlen <= EMBED_BATCH_SIZE) {
    //     return EMBED_BATCH_SIZE;
    // }
    // const int embed_batch_num = (seqlen + EMBED_BATCH_SIZE - 1) / EMBED_BATCH_SIZE;
    // const int embed_batch_num_per_chunk = (embed_batch_num + numa_num - 1) / numa_num;
    // assert(embed_batch_num_per_chunk >= 1);
    // return embed_batch_num_per_chunk * EMBED_BATCH_SIZE;
    return EMBED_BATCH_SIZE;
}

void ggml_graph_tp_processing(struct ggml_cgraph * cgraph) {
  // graph preprocessing
  // set embed_size and token_num for every node in cgraph
//  struct ggml_tensor *node0 = cgraph->nodes[0];
//  const size_t embed_size_  = node0->ne[0];
//  const size_t token_num    = node0->ne[1];
//  const size_t token_num    = EMBED_BATCH_SIZE;
//  set_mem_block(cgraph);

  total_layers_num = 0;


  size_t _batch_size;
  const int seqlen = cgraph->nodes[1]->ne[1];
// FIXME: adjust batch_size here
  if (cur_stage == PREFILL || cur_stage == SWITCHING_D2P) {
    // prefill stage
    _batch_size = compute_embed_mbatch_size(seqlen);
//    _batch_size = cgraph->nodes[0]->ne[1];
  } else {
    // decode stage
    _batch_size = seqlen; // only one embedding
  }
//  const size_t batch_size    = cgraph->nodes[0]->ne[1];
//  const size_t batch_size    = EMBED_BATCH_SIZE;
  const size_t batch_num     = 0; // FIXME: origin 0

//  reference_num_ ++;


  struct ggml_tensor *Kcur_i = NULL; /** = cgraph->nodes[KCUR_0_INDEX]->src[0]; **/
  struct ggml_tensor *Vcur_i_t = NULL; /** = cgraph->nodes[VCUR_0_T_INDEX]->src[0]; **/
  struct ggml_tensor *KQV = NULL;
  struct ggml_tensor *last_node = NULL;

//  // 将node0与node1融合为rms_norm
//  cgraph->nodes[1]->src[GGML_MAX_SRC-1] = cgraph->nodes[0];

  int last_node_num;

  for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {

    struct ggml_tensor *node = cgraph->nodes[node_n];
    set_embed_token(node, _batch_size, 0, MIN(_batch_size,cgraph->nodes[0]->ne[1]));

    last_node = node;
    last_node_num = node_n;

    if (node_n == cgraph->n_nodes - 1) {
      int t = node->op;
    }

    // set layer graph start and end
    if (node->op == GGML_OP_RMS_NORM && !strstr(node->src[0]->name, "ffn")) {
      // first node of a layer
      layer_graph_[total_layers_num].start = node;
      layer_graph_[total_layers_num].start_node_num = node_n;
    }
    if (node->op == GGML_OP_ADD && strstr(node->name, "l_out-")) {
      // last node of a layer
//      if (total_layers_num == 15) {
//        int t = node->op;
//      }
      layer_graph_[total_layers_num].end   = node;
      layer_graph_[total_layers_num].end_node_num   = node_n;
      total_layers_num ++;
    }

    // 确保在每一layer中获取正确的K和V
    if (node->op == GGML_OP_ROPE && strstr(node->name, "Kcur-")) {
//      total_layers_num ++;
      Kcur_i = node;
    }
    if (node->op == GGML_OP_MUL_MAT && strstr(node->name, "Vcur-")) {
      Vcur_i_t = node;
    }
    if (node->op == GGML_OP_MUL_MAT && strstr(node->name, "kqv")) {
      KQV = node;
    }

    // graph中无用node消除
    node->need_compute = 1;
    if (node->op == GGML_OP_GET_ROWS) {
      node->need_compute = 0;
    }
    if (node->op == GGML_OP_PERMUTE && strstr(node->name, "kqv")) {
      assert(KQV != NULL);
//      KQV->ne[0] = node->ne[0];
//      KQV->ne[1] = node->ne[1];
//      KQV->ne[2] = node->ne[2];
//      KQV->ne[3] = node->ne[3];
      node->need_compute = 0;
    }
    if (node->op == GGML_OP_CONT && strstr(node->name, "kqv")) {
      assert(KQV != NULL);
//      KQV->nb[0] = node->nb[0];
//      KQV->nb[1] = node->nb[1];
//      KQV->nb[2] = node->nb[2];
//      KQV->nb[3] = node->nb[3];
//      KQV->data = node->data;
      KQV->src[GGML_MAX_SRC-1] = node;
      node->src[GGML_MAX_SRC-1] = KQV;
      node->need_compute = 0;
    }
//     为GGML_OP_CPY设置正确的src，去除kv cache cpy的额外开销
    if (node->op == GGML_OP_CPY && strstr(node->name, "v_cache_view-")) {
      // Vcur_i_t : f32       v_cache : f16
      assert(Vcur_i_t != NULL);
      Vcur_i_t->data = node->data;
      // 调整V为v cache的内存布局，省去cpy
      Vcur_i_t->nb[0] = node->nb[0];
      Vcur_i_t->nb[1] = node->nb[1];
      Vcur_i_t->nb[2] = node->nb[2];
      Vcur_i_t->nb[3] = node->nb[3];
//      node->src[0] = Vcur_i_t;
      node->need_compute = 0;
    }
    if (node->op == GGML_OP_CPY && strstr(node->name, "k_cache_view-")) {
      // Kcur_i : f32         k_cache : f16
      assert(Kcur_i != NULL);
      Kcur_i->data = node->data;
      // 调整K为k cache的内存布局，省去cpy
      Kcur_i->nb[0] /= 2;
      Kcur_i->nb[1] /= 2;
      Kcur_i->nb[2] /= 2;
      Kcur_i->nb[3] /= 2;
//      node->src[0] = Kcur_i;
      node->need_compute = 0;
      Kcur_i->src[GGML_MAX_SRC-1] = node;
    }
    if (node->op == GGML_OP_VIEW && (strstr(node->name, "k_cache_view-") || strstr(node->name, "v_cache_view-"))) {
      node->need_compute = 0;
    }
    if (node->op == GGML_OP_TRANSPOSE && strstr(node->name, "Vcur-")) {
      node->need_compute = 0;
    }

    // TODO: 在k/v_cache_view中把src9设置为K/V, K需要为Kcur-i(reshaped)，这里没有设置成功
    if (node->op == GGML_OP_VIEW && strstr(node->name, "k-")) {
//      printf("e_k_ \n");
      assert(Kcur_i != NULL);
      node->src[GGML_MAX_SRC-1] = Kcur_i;
      // TODO: 在token并行中将Kcur-i的prep状态广播到k-i\ie_k_li (view)
      Kcur_i->src[GGML_MAX_SRC-1] = node;
    }
    if (node->op == GGML_OP_VIEW && strstr(node->name, "v-")) {
//      printf("e_v_\n");
      assert(Vcur_i_t != NULL);
      node->src[GGML_MAX_SRC-1] = Vcur_i_t;
      // TODO: 在token并行中将Vcur-i的prep状态广播到v-i\ie_k_li (view)
      Vcur_i_t->src[GGML_MAX_SRC-1] = node;
    }
//    if (node->op == GGML_OP_MUL_MAT && strstr(node->name, "kqv")) {
//      assert(KQV != NULL);
//      node->src[GGML_MAX_SRC-1] = KQV;
//      KQV->src[GGML_MAX_SRC-1] = node;
//    }

    if (node->op == GGML_OP_MUL_MAT) {
      node->mm_type = NORMAL_MM;
      if (strstr(node->name, "kq-")) {
        node->mm_type = KQ_MM;
      }
      if (strstr(node->name, "kqv-")) {
        node->mm_type = KQV_MM;
      }
      if (strstr(node->name, "Vcur-")) {
        // TODO: 改为计算Vcur转置，只需要对调src1与src0的位置，再对ne与nb进行修改即可
        struct ggml_tensor *tmp = node->src[0];
        node->src[0] = node->src[1];
        node->src[1] = tmp;

        int64_t ne_tmp = node->ne[0];
        node->ne[0] = node->ne[1];
        node->ne[1] = ne_tmp;

        node->nb[1] = node->nb[0] * node->ne[0];

        node->mm_type = Vt_MM;
      }
    }

    // set pre_nb for every tensor
    const int64_t ne0 = node->ne[0];
    const int64_t ne1 = node->ne[1];
    const int64_t ne2 = node->ne[2];
    const int64_t ne3 = node->ne[3];

    const int64_t prepared_num = ne1 * ne2 * ne3;
    node->prepared = static_cast<embed_prep_stat *>(malloc(prepared_num * sizeof(struct embed_prep_stat)));

    for (int64_t pi = 0; pi < prepared_num; pi++) {
      pthread_mutex_init(&node->prepared[pi].lck, NULL); // 初始化互斥锁
      node->prepared[pi].prepare_ = 0;
//        node->prepared[pi].cpy_done = 0;
    }

//    if (node_n == 0) {
//      // TODO: 将graph的第一个node的src设置为prepared
//      for (int64_t pi = 0; pi < prepared_num; pi++) {
//        node->prepared[pi].prepare_ = 1;
////        node->prepared[pi].cpy_done = 0;
//      }
//    } else {
//      for (int64_t pi = 0; pi < prepared_num; pi++) {
//        node->prepared[pi].prepare_ = 0;
////        node->prepared[pi].cpy_done = 0;
//      }
//    }

    node->pre_nb[0] = sizeof(struct embed_prep_stat); // embed_prep_stat for an embed
    node->pre_nb[1] = 1   * node->pre_nb[0];
    node->pre_nb[2] = ne1 * node->pre_nb[1];
    node->pre_nb[3] = ne2 * node->pre_nb[2];

    // FIXME: for debug
//      if (node->op == GGML_OP_MUL_MAT)
//        printf("%d: %s  op: %d\n", node_n, node->name, node->op);
    // FIXME: for debug
  }

  layer_graph_[total_layers_num].end   = last_node;
  layer_graph_[total_layers_num].end_node_num   = last_node_num;
}

#define NODE_INRANGE(n,l,r) ((n) >= (l) && (n) <= (r))
#define NODE_PER_LAYER      64
#define MEM_BLK_NUM         6

// mem_blk -> node_n
static const std::vector<int> node_mem_blk[NODE_PER_LAYER] = {
        [0] = {25, 26, 31},
        [1] = {1, 2, 20, 21, 23, 24},
        [2] = {3, 4, 6, 7, 12, 22, 27, 28, 30},
        [3] = {5, 17},
        [4] = {18, 19},
        [5] = {29},
};

static const std::vector<int> k_dst_node = {8, 10, 11, 16}; // k cache
static const std::vector<int> k_src_node = {10};
static const std::vector<int> v_dst_node = {9, 13, 14, 15}; // v cache
static const std::vector<int> v_src_node = {13};

static const int layer_param_node[] = {3, 6, 9, 23, 27, 29, 31};
static const int layer_param_src[]  = {0, 0, 1, 0, 0, 0, 0};

void write_layer_param(struct ggml_cgraph * cgraph) {
    const std::string dir         = get_layer_param_dir_c_str();
    const std::string config_file = get_layer_param_config_c_str();

    int ret = mkdir(dir.c_str(), 0755);
    if (ret != 0 && errno != EEXIST) {
        perror("mkdir failed when writing layer param!");
        exit(-1);
    }
    const int param_num = sizeof(layer_param_node) / sizeof(int);
    FILE *cf = fopen(config_file.c_str(), "wb");
    fseek(cf, 0, SEEK_SET);
    for (int subgraph_id = 0; subgraph_id < total_layers_num - 1; subgraph_id ++) {
        const int node_start_n = layer_graph_[subgraph_id].start_node_num;
        for (int p = 0; p < param_num; p++) {
            const int node_n = node_start_n + layer_param_node[p] - 1;
            const int src_n = layer_param_src[p];
            struct ggml_tensor * node = cgraph->nodes[node_n]->src[src_n];

            const std::string filename = dir + "/" + std::string(node->name);
            FILE *f = fopen(filename.c_str(), "wb");
            fseek(f, 0, SEEK_SET);
            fwrite(node->data, 1, node->nb[3], f);
            fclose(f);

            fprintf(cf, "%s\n", filename.c_str());
        }
    }
    fclose(cf);
}

static void alloc_param(struct memblk_info * mi, std::string &file, const int numa_node) {
    int fd = open(file.c_str(), O_RDONLY | O_DIRECT);
    lseek(fd, 0, SEEK_SET);

    // get file size
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        perror("fstat failed");
        close(fd);
        exit(-1);
    }

    size_t file_size = st.st_size;

    // mmap
    void * data;
    int ret;
    ret = posix_memalign(&data, PAGE_SIZE, file_size);
    if (ret != 0) {
        perror("posix_memalign");
        close(fd);
        exit(-1);
    }
//    madvise(data, file_size, MADV_NOHUGEPAGE);

    // mbind
    unsigned long nodemask = 1UL << numa_node;
    ret = mbind(data, file_size, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE_ALL);
    if (ret != 0) {
        fprintf(stderr, "mbind failed: %s\n", strerror(errno));
        close(fd);
        exit(-1);
    }

    assert(file_size % PAGE_SIZE == 0);
    ret = read(fd, data, file_size);
    if (ret < 0) {
        perror("read failed!");
        close(fd);
        exit(-1);
    }
    mlock(data, file_size);

    mi->size = file_size;
    mi->ptr = data;

    //    lseek(fd, 0, SEEK_SET);
    //    mi->fd = static_cast<int *>(malloc(REMAP_THREAD_NUM * sizeof(int)));
    //    mi->fd[0] = fd;
    //    for (int i = 1; i < REMAP_THREAD_NUM; i++) {
    //        mi->fd[i] = open(file.c_str(), O_RDONLY | O_DIRECT);
    //        assert(mi->fd[i] >= 0);
    //    }

    close(fd);

}

void load_layer_param() {
    int layer = 0;
    int param = 0;
    const std::string config = reinterpret_cast<const char *>(get_layer_param_config_c_str());
    std::ifstream cf(config);
    std::string filename;

    //    unsigned long nodemask = 1UL << (layer % numa_num);
    //    int ret = set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
    //    if (ret != 0) {
    //        fprintf(stderr, "set_mempolicy failed: %s\n", strerror(errno));
    //        exit(-1);
    //    }

//    reset_open_file_limit();

    while (std::getline(cf, filename)) {
        // layer_param_filename[layer][param] = filename;
        alloc_param(&layer_param_info[layer][param], filename, layer % numa_num);
        //        printf("param: %d, layer: %d, numa node: %d\n", param, layer, layer % numa_num);
        param ++;
        if (param == LAYER_PARAM_NUM) { // move to next layer
            param = 0;
            layer ++;
            //            nodemask = 1UL << (layer % numa_num);
            //            ret = set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
            //            if (ret != 0) {
            //                fprintf(stderr, "set_mempolicy failed: %s\n", strerror(errno));
            //                exit(-1);
            //            }
        }
    }
    cf.close();
}

int       remap_thread_num;
pthread_t remap_threads[MAX_REMAP_THREAD_NUM];

static void remap_param_p2d(struct memblk_info * mi, const int cur_numa_node, const int ith, const int nth) {
    const size_t file_size = mi->size;
    void * data = mi->ptr;
//    int fd = mi->fd[ith];

    void *addrs[INTERLEAVE_GAP];
    int nodes[INTERLEAVE_GAP];
    int status[INTERLEAVE_GAP];

    assert(file_size % PAGE_SIZE == 0);
    const size_t page_num = file_size / PAGE_SIZE;
    const size_t group_num = (page_num + INTERLEAVE_GAP - 1) / INTERLEAVE_GAP;
    const size_t group_size = INTERLEAVE_GAP * PAGE_SIZE;
    for (size_t page_group = ith; page_group < group_num; page_group += nth) {
        const int numa_node = page_group % numa_num;
        unsigned long nodemask = 1UL << numa_node;
        if (numa_node == cur_numa_node) {
            continue;
        }

        void * page_group_addr = (void *)((char *)data + page_group * group_size);
        const size_t page_n = std::min((size_t)INTERLEAVE_GAP, page_num - page_group * INTERLEAVE_GAP);
        const unsigned long page_group_size = page_n * PAGE_SIZE;

        const size_t group_page_num = page_group_size / PAGE_SIZE;
        for (size_t p_i = 0; p_i < group_page_num; p_i ++) {
            addrs[p_i] = (char *)page_group_addr + p_i * PAGE_SIZE;
            nodes[p_i] = numa_node;
        }

//        munlock(page_group_addr, page_group_size);
        int ret = move_pages(0, group_page_num, addrs, nodes, status, 0);
        if (ret == -1) {
            perror("move_pages query after move failed");
            exit(-1);
        }

        for (int s = 0; s < group_page_num; s++) {
            assert(status[s] == numa_node);
        }

        /**
        madvise(page_group_addr, page_group_size, MADV_DONTNEED);
        int ret = mbind(page_group_addr, page_group_size, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE_ALL);
        if (ret != 0) {
            fprintf(stderr, "mbind failed: %s\n", strerror(errno));
            exit(-1);
        }
         **/

//        lseek(fd, page_group * group_size, SEEK_SET);
//        int read_bytes = read(fd, page_group_addr, page_group_size);
//        if (read_bytes == -1) {
//            perror("read failed! ");
//            exit(-1);
//        }

        /**
        const size_t group_page_num = page_group_size / PAGE_SIZE;
        for (size_t p_i = 0; p_i < group_page_num; p_i ++) {
            volatile char c = *((char *)page_group_addr + PAGE_SIZE * p_i);
            memset((char *)page_group_addr + PAGE_SIZE * p_i, 0, 1); // write to page, make sure page hard copy
            memset((char *)page_group_addr + PAGE_SIZE * p_i, c, 1);
        }
         **/
        mlock(page_group_addr, page_group_size);
    }
}

static void remap_param_d2p(struct memblk_info * mi, const int cur_numa_node, const int ith, const int nth) {
    const size_t file_size = mi->size;
    void * data = mi->ptr;
    //    int fd = mi->fd[ith];

    void *addrs[INTERLEAVE_GAP];
    int nodes[INTERLEAVE_GAP];
    int status[INTERLEAVE_GAP];

    assert(file_size % PAGE_SIZE == 0);
    const size_t page_num = file_size / PAGE_SIZE;
    const size_t group_num = (page_num + INTERLEAVE_GAP - 1) / INTERLEAVE_GAP;
    const size_t group_size = INTERLEAVE_GAP * PAGE_SIZE;
    for (size_t page_group = ith; page_group < group_num; page_group += nth) {
        const int numa_node = page_group % numa_num;
        unsigned long nodemask = 1UL << numa_node;
        if (numa_node == cur_numa_node) {
            continue;
        }

        void * page_group_addr = (void *)((char *)data + page_group * group_size);
        const size_t page_n = std::min((size_t)INTERLEAVE_GAP, page_num - page_group * INTERLEAVE_GAP);
        const unsigned long page_group_size = page_n * PAGE_SIZE;

        const size_t group_page_num = page_group_size / PAGE_SIZE;
        for (size_t p_i = 0; p_i < group_page_num; p_i ++) {
            addrs[p_i] = (char *)page_group_addr + p_i * PAGE_SIZE;
            nodes[p_i] = cur_numa_node;
        }

//        munlock(page_group_addr, page_group_size);
        int ret = move_pages(0, group_page_num, addrs, nodes, status, 0);
        if (ret == -1) {
            perror("move_pages query after move failed");
            exit(-1);
        }

        for (int s = 0; s < group_page_num; s++) {
            assert(status[s] == cur_numa_node);
        }

        mlock(page_group_addr, page_group_size);
    }
}

static bool last_layer_infer_done;
static int  remap_layer_num_per_batch;

static void init_remap_layer_num_per_batch() {
    const int avg_layer = (total_layers_num - 1 + SLIDE_WINDOW_SIZE - 1) / SLIDE_WINDOW_SIZE;
    remap_layer_num_per_batch = std::max(avg_layer, MIN_REMAP_BATHC_LAYER);
    // remap_layer_num_per_batch = total_layers_num - 1;
}

inline static bool if_remap_proceed(const bool is_first_remap_layer) {
    if (!is_first_remap_layer && last_layer_infer_done) {
        return false;
    }
    return true;
}

// TODO: multi_thread, async
static void *remap_model_param_p2d_thread(void *ith_) { // in(int) : {nth: 16 / ith: 16 }
    const int ith = *(int *)ith_;
    const int param_num = sizeof(layer_param_node) / sizeof(int);

    bool is_first_remap_layer = true;
    const int64_t cur_switch_end_layer = std::min(cur_switch_start_layer + remap_layer_num_per_batch, total_layers_num -1);
    for (int subgraph_id = cur_switch_start_layer; subgraph_id < cur_switch_end_layer; subgraph_id ++) {
//        if (ith == 0) {
//            last_layer_infer_done = layer_infer_done[total_layers_num-2];
//        }
//        ggml_barrier_remap();
//
//        if (!if_remap_proceed(is_first_remap_layer)) {
//            break;
//        }
//        ggml_barrier_remap();
//
//        if (is_first_remap_layer) {
//            is_first_remap_layer = false;
//        }

        // remap every param in current layer
        for (int p = 0; p < param_num; p++) {
            const int cur_numa_node = subgraph_id % numa_num;
            remap_param_p2d(&layer_param_info[subgraph_id][p], cur_numa_node, ith, REMAP_THREAD_NUM);
        }
        if (ith == 0) {
            cur_switch_start_layer ++;
            cur_switch_remain_layer --;
        }
    }
    return NULL;
}

static void *remap_model_param_d2p_thread(void *ith_) { // in(int) : {nth: 16 / ith: 16 }
    const int ith = *(int *)ith_;
    const int param_num = sizeof(layer_param_node) / sizeof(int);

    bool is_first_remap_layer = true;
    const int64_t cur_switch_end_layer = std::min(cur_switch_start_layer + remap_layer_num_per_batch, total_layers_num -1);
    for (int subgraph_id = cur_switch_start_layer; subgraph_id < cur_switch_end_layer; subgraph_id ++) {
//        if (ith == 0) {
//            last_layer_infer_done = layer_infer_done[total_layers_num-2];
//        }
//        ggml_barrier_remap();
//
//        if (!if_remap_proceed(is_first_remap_layer)) {
//            break;
//        }
//        ggml_barrier_remap();
//
//        if (is_first_remap_layer) {
//            is_first_remap_layer = false;
//        }

        // remap every param in current layer
        for (int p = 0; p < param_num; p++) {
            const int cur_numa_node = subgraph_id % numa_num;
            remap_param_d2p(&layer_param_info[subgraph_id][p], cur_numa_node, ith, REMAP_THREAD_NUM);
        }
        if (ith == 0) {
            cur_switch_start_layer ++;
            cur_switch_remain_layer --;
        }
    }
    return NULL;
}

static int thread_id[MAX_REMAP_THREAD_NUM];

static void remap_model_param_p2d_start() {
    for (int ith = 0; ith < REMAP_THREAD_NUM; ++ith) {
        thread_id[ith] = ith;
        int ret = pthread_create(&remap_threads[ith], NULL, remap_model_param_p2d_thread, (void *)&thread_id[ith]);
        assert(ret == 0);  // 确保线程创建成功
    }
}

static void remap_model_param_p2d_end() {
    for (int i = 0; i < REMAP_THREAD_NUM; ++i) {
        pthread_join(remap_threads[i], NULL);
    }
}

static void remap_model_param_d2p_start() {
    for (int ith = 0; ith < REMAP_THREAD_NUM; ++ith) {
        thread_id[ith] = ith;
        int ret = pthread_create(&remap_threads[ith], NULL, remap_model_param_d2p_thread, (void *)&thread_id[ith]);
        assert(ret == 0);  // 确保线程创建成功
    }
}

static void remap_model_param_d2p_end() {
    for (int i = 0; i < REMAP_THREAD_NUM; ++i) {
        pthread_join(remap_threads[i], NULL);
    }
}


static void realloc_numa_mem_(struct ggml_cgraph * cgraph) {
    cgraph->nodes[0]->data = numa_mem_info[0][0].ptr; // set data for node00;

    for (int il = 0; il < total_layers_num - 1; il ++) {
        // 遍历每一个子图，分配numa内存
        const int numa_node = il % numa_num;
        const int start_node_n = layer_graph_[il].start_node_num;
        const int end_node_n   = layer_graph_[il].end_node_num;
        for (int m = 0; m < MEM_BLK_NUM; m++) {
            const auto &node_list = node_mem_blk[m];
            const size_t node_num = node_list.size();
            for (size_t n = 0; n < node_num; n++) {
                const int node_off_n = node_list[n] - 1;
                assert(start_node_n + node_off_n <= end_node_n);
                struct ggml_tensor * node = cgraph->nodes[start_node_n + node_off_n];
                assert(node->nb[3] <= numa_mem_info[numa_node][m].size);
                node->data = numa_mem_info[numa_node][m].ptr;
            }
        }

        // layer output
        const int last_node_off_n = 31;
        struct ggml_tensor * l_out = cgraph->nodes[start_node_n + last_node_off_n];
        l_out->data = numa_l_out_info[il].ptr;

        // k cache
        const void * k_cache_ = k_cache_info[il].ptr;
        for (auto k_d : k_dst_node) {
            const int node_off_n = k_d - 1;
            struct ggml_tensor * node = cgraph->nodes[start_node_n + node_off_n];
            node->data = (void *)k_cache_;
        }
        for (auto k_s : k_src_node) {
            const int node_off_n = k_s - 1;
            struct ggml_tensor * node = cgraph->nodes[start_node_n + node_off_n]->src[0];
            node->data = (void *)k_cache_;
        }

        // v cache
        const void * v_cache_ = v_cache_info[il].ptr;
        for (auto v_d : v_dst_node) {
            const int node_off_n = v_d - 1;
            struct ggml_tensor * node = cgraph->nodes[start_node_n + node_off_n];
            node->data = (void *)v_cache_;
        }
        for (auto v_s : v_src_node) {
            const int node_off_n = v_s - 1;
            struct ggml_tensor * node = cgraph->nodes[start_node_n + node_off_n]->src[0];
            node->data = (void *)v_cache_;
        }

        // layer param
        const int param_num = sizeof(layer_param_node) / sizeof(int);
        const int node_start_n = layer_graph_[il].start_node_num;
        for (int p = 0; p < param_num; p++) {
            const int node_n = node_start_n + layer_param_node[p] - 1;
            const int src_n = layer_param_src[p];
            struct ggml_tensor * node = cgraph->nodes[node_n]->src[src_n];
            node->data = layer_param_info[il][p].ptr;
        }
    }
}

static void printf_layer(int layer, struct ggml_cgraph * cgraph, const char *file) {
    FILE *f = fopen(file, "wb");
    const int start_node_off = layer_graph_[layer].start_node_num;
    const int end_node_off = layer_graph_[layer].end_node_num;
    const int max_src_num = 3;
    for (int node_n = start_node_off; node_n <= end_node_off; node_n ++) {
        struct ggml_tensor *node = cgraph->nodes[node_n];
        fprintf(f, "%-40s ---- 0x%016lx   ----   nb3: %-10ld\n", node->name, (unsigned long)node->data, node->nb[3]);
        for (int s = 0; s < max_src_num; s ++) {
             struct ggml_tensor *src = cgraph->nodes[node_n]->src[s];
             if (src != nullptr) {
                 fprintf(f, "%-40s ---- 0x%016lx   ----   nb3: %-10ld\n", src->name, (unsigned long)src->data, src->nb[3]);
             } else {
                 fprintf(f, "NULL\n");
             }
        }
        struct ggml_tensor *src = cgraph->nodes[node_n]->src[9];
        if (src != nullptr) {
            fprintf(f, "%-40s ---- 0x%016lx   ----   nb3: %-10ld\n", src->name, (unsigned long)src->data, src->nb[3]);
        } else {
            fprintf(f, "NULL\n");
        }
        fprintf(f, "----------------------------------------------------------\n");
    }
    fprintf(f, "\n==========================================================\n\n");
    fflush(f);
    fclose(f);
}

// Fitted TPS function for LLaMA-1B (no-switch scenario)
// y(x) = {
//   1.623103 * x + 7.103103,        when x ≤ 64
//   21.196127 * ln(x) + 24.475677,  when 64 < x ≤ 128
//   128,                            when x > 128
// }
static float fit_llama1B_sp_no_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input
    float fit_tps;
    if (batch_size <= 64) {
        fit_tps = 1.623103f * batch_size + 7.103103f;  // linear region
    } else if (batch_size <= 128) {
        fit_tps = 21.196127f * std::log((float)batch_size) + 24.475677f;  // log region
    } else {
        fit_tps = 128.0f;  // saturation region
    }
    return fit_tps;
}

// Fitted TPS function for LLaMA-1B (switch scenario)
// y(x) = {
//   1.215474 * x + 5.913520,        when x ≤ 64
//   18.803098 * ln(x) + 5.017678,   when 64 < x ≤ 128
//   97.775000,                      when x > 128
// }
static float fit_llama1B_sp_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input
    float fit_tps;
    if (batch_size <= 64) {
        fit_tps = 1.215474f * batch_size + 5.913520f;  // linear region
    } else if (batch_size <= 128) {
        fit_tps = 18.803098f * std::log((float)batch_size) + 5.017678f;  // log region
    } else {
        fit_tps = 97.775f;  // saturation region
    }
    return fit_tps;
}


// Fitted TPS function for LLaMA-1B (TP, switch scenario)
// y(x) = {
//   2.402522 * x + 1.233043,                               when x ≤ 8
//   28.798744 * ln(x) - 45.692178,                         when 8 < x ≤ 64
//   x / (0.00001105 * x^2 + 0.004361 * x + 0.514933),      when x ≥ 64
// }
static float fit_llama1B_tp_switch(const int batch_size) {
    assert(batch_size > 0 && batch_size <= BATCH_SIZE_LIMIT);  // ensure valid input
    float fit_tps;

    if (batch_size <= 8) {
        // Linear region: y = 2.402522 * x + 1.233043
        fit_tps = 2.402522f * batch_size + 1.233043f;
    } else if (batch_size <= 64) {
        // Logarithmic region: y = 28.798744 * ln(x) - 45.692178
        fit_tps = 28.798744f * std::log((float)batch_size) - 45.692178f;
    } else {
        // Rational region: y = x / (0.00001105 * x^2 + 0.004361 * x + 0.514933)
        float x = static_cast<float>(batch_size);
        fit_tps = x / (0.00001105f * x * x + 0.004361f * x + 0.514933f);
    }

    return fit_tps;
}


// Fitted TPS function for LLaMA-1B (TP, no-switch scenario)
// y(x) = {
//   6.376696 * x + 5.447391,                               when x ≤ 8
//   17.860028 * ln(x) + 15.518771,                         when 8 < x ≤ 96
//   x / (0.00000853 * x^2 + 0.004906 * x + 0.344604),      when x ≥ 96
// }
static float fit_llama1B_tp_no_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input
    float fit_tps;

    if (batch_size <= 8) {
        fit_tps = 6.376696f * batch_size + 5.447391f;  // linear region
    } else if (batch_size <= 96) {
        fit_tps = 17.860028f * std::log((float)batch_size) + 15.518771f;  // log region
    } else {
        float x = static_cast<float>(batch_size);
        fit_tps = x / (0.00000853f * x * x + 0.004906f * x + 0.344604f);  // rational region
    }

    return fit_tps;
}


// Fitted TPS function (TP, no-switch scenario)
// y(x) = {
//   1.590435 * x + 4.100870,                               when x ≤ 8
//   5.690543 * ln(x) + 3.393443,                           when 8 < x ≤ 64
//   x / (0.00001028 * x^2 + 0.020642 * x + 1.020348),      when x ≥ 64
// }
static float fit_llama3B_tp_no_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input
    float fit_tps;

    if (batch_size <= 8) {
        fit_tps = 1.590435f * batch_size + 4.100870f;  // linear region
    } else if (batch_size <= 64) {
        fit_tps = 5.690543f * std::log((float)batch_size) + 3.393443f;  // log region
    } else {
        float x = static_cast<float>(batch_size);
        fit_tps = x / (0.00001028f * x * x + 0.020642f * x + 1.020348f);  // rational region
    }

    return fit_tps;
}


// Fitted TPS function (TP, switch scenario)
// y(x) = {
//   1.018348 * x + 0.508696,                               when x ≤ 8
//   7.790441 * ln(x) - 8.691838,                           when 8 < x ≤ 64
//   x / (0.00002706 * x^2 + 0.015086 * x + 1.728531),      when x ≥ 64
// }
static float fit_llama3B_tp_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input
    float fit_tps;

    if (batch_size <= 8) {
        fit_tps = 1.018348f * batch_size + 0.508696f;  // linear region
    } else if (batch_size <= 64) {
        fit_tps = 7.790441f * std::log((float)batch_size) - 8.691838f;  // log region
    } else {
        float x = static_cast<float>(batch_size);
        fit_tps = x / (0.00002706f * x * x + 0.015086f * x + 1.728531f);  // rational region
    }

    return fit_tps;
}


// Fitted TPS function (PP, no-switch scenario)
// y(x) = {
//   3.563457 * ln(x) + 4.823333,                          when 2 ≤ x ≤ 8
//   12.416667,                                            when 8 < x ≤ 16
//   0.639107 * x + 3.120000,                              when 16 < x ≤ 64
//   1.426822 * ln(x) + 37.896841,                         when 64 < x ≤ 128
//   44.765000,                                            when x > 128
// }
static float fit_llama3B_sp_no_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input (given domain starts at 2)
    float fit_tps;

    if (batch_size <= 8) {
        fit_tps = 3.563457f * std::log((float)batch_size) + 4.823333f;  // log region
    } else if (batch_size <= 16) {
        fit_tps = 12.416667f;  // constant region
    } else if (batch_size <= 64) {
        fit_tps = 0.639107f * batch_size + 3.120000f;  // linear region
    } else if (batch_size <= 128) {
        fit_tps = 1.426822f * std::log((float)batch_size) + 37.896841f;  // log region
    } else {
        fit_tps = 44.765000f;  // constant region
    }

    return fit_tps;
}


// Fitted TPS function (PP, switch scenario)
// y(x) = {
//   5.243546 * ln(x) - 1.367043,                           when 2 ≤ x ≤ 16
//   13.790000,                                             when x == 16   (constant region)
//   0.188884 * x + 11.475000,                              when 16 < x ≤ 64
//   13.483923 * ln(x) - 34.195929,                         when 64 < x ≤ 128
//   33.935000,                                             when x > 128
// }
static float fit_llama3B_sp_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input (given domain starts at 2)
    float fit_tps;

    if (batch_size <= 16) {
        fit_tps = 5.243546f * std::log((float)batch_size) - 1.367043f;  // log region
        if (batch_size == 16) {
            fit_tps = 13.790000f;  // override with constant value at boundary
        }
    } else if (batch_size <= 64) {
        fit_tps = 0.188884f * batch_size + 11.475000f;  // linear region
    } else if (batch_size <= 128) {
        fit_tps = 13.483923f * std::log((float)batch_size) - 34.195929f;  // log region
    } else {
        fit_tps = 33.935000f;  // constant region
    }

    return fit_tps;
}


// Fitted TPS function (TP, no-switch scenario)
// y(x) = {
//   0.938348 * x + 2.108696,                               when x ≤ 8
//   2.944895 * ln(x) + 3.098382,                           when 8 < x ≤ 64
//   x / (-0.00000409 * x^2 + 0.046568 * x + 1.269246),     when x ≥ 64
// }
static float fit_llama8B_tp_no_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input
    float fit_tps;

    if (batch_size <= 8) {
        fit_tps = 0.938348f * batch_size + 2.108696f;  // linear region
    } else if (batch_size <= 64) {
        fit_tps = 2.944895f * std::log((float)batch_size) + 3.098382f;  // log region
    } else {
        float x = static_cast<float>(batch_size);
        fit_tps = x / (-0.00000409f * x * x + 0.046568f * x + 1.269246f);  // rational region
    }

    return fit_tps;
}


// Fitted TPS function (TP, switch scenario)
// y(x) = {
//   0.423217 * x + 0.170435,                               when x ≤ 8
//   4.229347 * ln(x) - 5.995734,                           when 8 < x ≤ 64
//   x / (-0.00001830 * x^2 + 0.051527 * x + 2.301704),     when x ≥ 64
// }
static float fit_llama8B_tp_switch(const int batch_size) {
    assert(batch_size > 0);  // ensure valid input
    float fit_tps;

    if (batch_size <= 8) {
        fit_tps = 0.423217f * batch_size + 0.170435f;  // linear region
    } else if (batch_size <= 64) {
        fit_tps = 4.229347f * std::log((float)batch_size) - 5.995734f;  // log region
    } else {
        float x = static_cast<float>(batch_size);
        fit_tps = x / (-0.00001830f * x * x + 0.051527f * x + 2.301704f);  // rational region
    }

    return fit_tps;
}


// Fitted TPS function (PP, no-switch scenario)
// y(x) = {
//   1.478762 * ln(x) + 2.400000,                          when 2 ≤ x ≤ 8
//   5.733333,                                            when 8 < x ≤ 16
//   0.308170 * x + 1.285000,                             when 16 < x ≤ 64
//   1.804291 * ln(x) + 13.495424,                        when 64 < x ≤ 128
//   22.290000,                                           when x > 128
// }
static float fit_llama8B_sp_no_switch(const int batch_size) {
    assert(batch_size >= 2);  // ensure valid input (given domain starts at 2)
    float fit_tps;

    if (batch_size <= 8) {
        fit_tps = 1.478762f * std::log((float)batch_size) + 2.400000f;  // log region
    } else if (batch_size <= 16) {
        fit_tps = 5.733333f;  // constant region
    } else if (batch_size <= 64) {
        fit_tps = 0.308170f * batch_size + 1.285000f;  // linear region
    } else if (batch_size <= 128) {
        fit_tps = 1.804291f * std::log((float)batch_size) + 13.495424f;  // log region
    } else {
        fit_tps = 22.290000f;  // constant region
    }

    return fit_tps;
}


// Fitted TPS function (PP, switch scenario)
// y(x) = {
//   1.426348 * ln(x) - 0.108206,                         when 2 ≤ x ≤ 16
//   3.950000,                                           when x == 16 (constant region)
//   0.222143 * x + 0.410000,                            when 16 < x ≤ 64
//   4.266870 * ln(x) - 3.204626,                        when 64 < x ≤ 128
//   18.565000,                                          when x > 128
// }
static float fit_llama8B_sp_switch(const int batch_size) {
    assert(batch_size >= 2);  // ensure valid input (given domain starts at 2)
    float fit_tps;

    if (batch_size < 16) {
        fit_tps = 1.426348f * std::log((float)batch_size) - 0.108206f;  // log region
    } else if (batch_size == 16) {
        fit_tps = 3.950000f;  // constant at boundary
    } else if (batch_size <= 64) {
        fit_tps = 0.222143f * batch_size + 0.410000f;  // linear region
    } else if (batch_size <= 128) {
        fit_tps = 4.266870f * std::log((float)batch_size) - 3.204626f;  // log region
    } else {
        fit_tps = 18.565000f;  // constant region
    }

    return fit_tps;
}


static float compute_llama1B_sp_no_switch_time(const int batch_size, const int window_num) {
    const float tps = fit_llama1B_sp_no_switch(batch_size);
    return batch_size * SLIDE_WINDOW_SIZE * window_num / tps;
}

static float compute_llama1B_sp_switch_time(const int batch_size, const int window_num) {
    const float tps_no_switch = fit_llama1B_sp_no_switch(batch_size);
    const float tps_switch = fit_llama1B_sp_switch(batch_size);

    const int switch_round = (total_layers_num - 1 + remap_layer_num_per_batch - 1) / remap_layer_num_per_batch;
    assert(switch_round <= SLIDE_WINDOW_SIZE);
    const int no_switch_round = window_num * SLIDE_WINDOW_SIZE - switch_round;

    return (batch_size * switch_round / tps_switch) + (batch_size * no_switch_round / tps_no_switch);
}

static float compute_llama1B_tp_no_switch_time(const int batch_size, const int window_num) {
    const float tps = fit_llama1B_tp_no_switch(batch_size);
    return batch_size * SLIDE_WINDOW_SIZE * window_num / tps;
}

static float compute_llama1B_tp_switch_time(const int batch_size, const int window_num) {
    const float tps_no_switch = fit_llama1B_tp_no_switch(batch_size);
    const float tps_switch = fit_llama1B_tp_switch(batch_size);

    const int switch_round = (total_layers_num - 1 + remap_layer_num_per_batch - 1) / remap_layer_num_per_batch;
    assert(switch_round <= SLIDE_WINDOW_SIZE);
    const int no_switch_round = window_num * SLIDE_WINDOW_SIZE - switch_round;

    return (batch_size * switch_round / tps_switch) + (batch_size * no_switch_round / tps_no_switch);
}


static float compute_llama3B_sp_no_switch_time(const int batch_size, const int window_num) {
    const float tps = fit_llama3B_sp_no_switch(batch_size);
    return batch_size * SLIDE_WINDOW_SIZE * window_num / tps;
}

static float compute_llama3B_sp_switch_time(const int batch_size, const int window_num) {
    const float tps_no_switch = fit_llama3B_sp_no_switch(batch_size);
    const float tps_switch = fit_llama3B_sp_switch(batch_size);

    const int switch_round = (total_layers_num - 1 + remap_layer_num_per_batch - 1) / remap_layer_num_per_batch;
    assert(switch_round <= SLIDE_WINDOW_SIZE);
    const int no_switch_round = window_num * SLIDE_WINDOW_SIZE - switch_round;

    return (batch_size * switch_round / tps_switch) + (batch_size * no_switch_round / tps_no_switch);
}

static float compute_llama3B_tp_no_switch_time(const int batch_size, const int window_num) {
    const float tps = fit_llama3B_tp_no_switch(batch_size);
    return batch_size * SLIDE_WINDOW_SIZE * window_num / tps;
}

static float compute_llama3B_tp_switch_time(const int batch_size, const int window_num) {
    const float tps_no_switch = fit_llama3B_tp_no_switch(batch_size);
    const float tps_switch = fit_llama3B_tp_switch(batch_size);

    const int switch_round = (total_layers_num - 1 + remap_layer_num_per_batch - 1) / remap_layer_num_per_batch;
    assert(switch_round <= SLIDE_WINDOW_SIZE);
    const int no_switch_round = window_num * SLIDE_WINDOW_SIZE - switch_round;

    return (batch_size * switch_round / tps_switch) + (batch_size * no_switch_round / tps_no_switch);
}

static float compute_llama8B_sp_no_switch_time(const int batch_size, const int window_num) {
    const float tps = fit_llama8B_sp_no_switch(batch_size);
    return batch_size * SLIDE_WINDOW_SIZE * window_num / tps;
}

static float compute_llama8B_sp_switch_time(const int batch_size, const int window_num) {
    const float tps_no_switch = fit_llama8B_sp_no_switch(batch_size);
    const float tps_switch = fit_llama8B_sp_switch(batch_size);

    const int switch_round = (total_layers_num - 1 + remap_layer_num_per_batch - 1) / remap_layer_num_per_batch;
    assert(switch_round <= SLIDE_WINDOW_SIZE);
    const int no_switch_round = window_num * SLIDE_WINDOW_SIZE - switch_round;

    return (batch_size * switch_round / tps_switch) + (batch_size * no_switch_round / tps_no_switch);
}

static float compute_llama8B_tp_no_switch_time(const int batch_size, const int window_num) {
    const float tps = fit_llama8B_tp_no_switch(batch_size);
    return batch_size * SLIDE_WINDOW_SIZE * window_num / tps;
}

static float compute_llama8B_tp_switch_time(const int batch_size, const int window_num) {
    const float tps_no_switch = fit_llama8B_tp_no_switch(batch_size);
    const float tps_switch = fit_llama8B_tp_switch(batch_size);

    const int switch_round = (total_layers_num - 1 + remap_layer_num_per_batch - 1) / remap_layer_num_per_batch;
    assert(switch_round <= SLIDE_WINDOW_SIZE);
    const int no_switch_round = window_num * SLIDE_WINDOW_SIZE - switch_round;

    return (batch_size * switch_round / tps_switch) + (batch_size * no_switch_round / tps_no_switch);
}

struct win_node {
    struct win_node * prev;
    struct win_node * next;
    int win_avg_batch_size;
};

static std::vector<float>   win_batch_weight = {0.1f, 0.2f, 0.3f, 0.4f};
static std::vector<float>   estimate_error_threshold = {0.15f, 0.25f, 0.35f, 0.45f};
static int32_t              estimate_absolute_error_threshold = 8;
static struct win_node      win_node_mem[SLIDE_WINDOW_NUM];
static struct win_node *    win_head;
static struct win_node *    win_tail;
static int                  cur_batch_size;
static int                  old_estimate_batch_size;

static void init_slide_window(const int batch_size) {
    win_head = &win_node_mem[0];
    win_tail = &win_node_mem[SLIDE_WINDOW_NUM - 1];
    win_node_mem[0].win_avg_batch_size = batch_size;
    win_node_mem[0].prev = &win_node_mem[SLIDE_WINDOW_NUM - 1];
    win_node_mem[0].next = &win_node_mem[1];
    for (int i = 1; i < SLIDE_WINDOW_NUM; i++) {
        auto *node = &win_node_mem[i];
        node->win_avg_batch_size = 1;
        node->prev = &win_node_mem[(i - 1) % SLIDE_WINDOW_NUM];
        node->next = &win_node_mem[(i + 1) % SLIDE_WINDOW_NUM];
    }
    cur_batch_size = 0;
    old_estimate_batch_size = batch_size;
}

static void add_to_window(const int avg_batch_size) {
    win_tail = win_head;
    win_head = win_head->next;
    win_tail->win_avg_batch_size = avg_batch_size;
}

static int estimate_future_batch_size() {
    float future_batch_size = 0;
    auto *cur_node = win_head;
    for (int i = 0; i < SLIDE_WINDOW_NUM; i++) {
        future_batch_size += win_batch_weight[i] * cur_node->win_avg_batch_size;
        cur_node = cur_node->next;
    }
    return future_batch_size;
}

static float compute_estimate_error(const int estimate, const int real) {
    return ((float)std::abs(real - estimate)) / real;
}

static int compute_future_window_num(const int estimate, const int real) {
    int retval = 0;
    const float estimate_error = compute_estimate_error(estimate, real);
    if (estimate_error <= estimate_error_threshold[0]) {
        retval = SLIDE_WINDOW_NUM;
    } else if (estimate_error <= estimate_error_threshold[1]) {
        retval = SLIDE_WINDOW_NUM - 1;
    } else if (estimate_error <= estimate_error_threshold[2]) {
        retval = SLIDE_WINDOW_NUM - 2;
    } else {
        retval = SLIDE_WINDOW_NUM - 3;
    }

    if (std::abs(estimate - real) < estimate_absolute_error_threshold) {
        retval = SLIDE_WINDOW_NUM;
    }
    return retval;
}

static bool need_move_to_new_window(const int r) {
    if (r > 0 && r % SLIDE_WINDOW_SIZE == 0) {
        return true;
    }
    return false;
}

// FIXME: using fancy mathmatics to judge which phase to use in next round inference
// TODO: return SWITCHING when switch is needed, otherwise return stage according to reference_num_(r)
static enum InferStage compute_cur_stage(int r, const int batch_size) {
    enum InferStage new_stage;
    assert(r >= 0);

    if (need_move_to_new_window(r)) { // move to next window
        // estimate new b^
        // estimate new batch round
        // decide new stage according to current stage
        const int avg_batch_size = cur_batch_size / SLIDE_WINDOW_SIZE;
        add_to_window(avg_batch_size);

        const int future_estimate_batch_size = estimate_future_batch_size();
        const int future_window = compute_future_window_num(old_estimate_batch_size, avg_batch_size);
        old_estimate_batch_size = future_estimate_batch_size;
        cur_batch_size = batch_size;

        // use future_window and future_estimate_batch_size to decide new stage
        assert(cur_stage != SWITCHING_P2D && cur_stage != SWITCHING_D2P);
        if (cur_stage == PREFILL) {
            const float no_switch_time = compute_llama8B_sp_no_switch_time(future_estimate_batch_size, future_window);
            const float switch_time = compute_llama8B_tp_switch_time(future_estimate_batch_size, future_window);
            if (no_switch_time < switch_time) {
                new_stage = PREFILL;
            } else {
                new_stage = SWITCHING_P2D;
            }
        }
        if (cur_stage == DECODE) {
            const float no_switch_time = compute_llama8B_tp_no_switch_time(future_estimate_batch_size, future_window);
            const float switch_time = compute_llama8B_sp_switch_time(future_estimate_batch_size, future_window);
            if (no_switch_time < switch_time) {
                new_stage = DECODE;
            } else {
                new_stage = SWITCHING_D2P;
            }
        }

    } else { // still in current window
        cur_batch_size += batch_size;
        new_stage = cur_stage;
    }

    return new_stage;
}

static enum InferStage compute_cur_stage_test(int r, const int batch_size) {
    // if (r > 0 && r < 9) {
    //     return SWITCHING_P2D;
    // }
    // return DECODE;
    return PREFILL;
}

static int is_first_round = 1;

enum SwitchStage     cur_switch_stage;
int64_t              cur_switch_start_layer;
int64_t              cur_switch_remain_layer;

static void init_layer_infer_done() {
    for (int i = 0; i < LAYER_UPPER_BOUND; i++) {
        layer_infer_done[i] = 0;
    }
}

static void set_decode_chunk_size() {
    switch (total_layers_num) {
        case LLamaModelLayerNum::ln_1B:
            decode_chunk_size_ = LlamaModelDecodeChunk::dc_1B;
            break;
        case LLamaModelLayerNum::ln_3B:
            decode_chunk_size_ = LlamaModelDecodeChunk::dc_3B;
            break;
        case LLamaModelLayerNum::ln_8B:
            decode_chunk_size_ = LlamaModelDecodeChunk::dc_8B;
            break;
        default:
            fprintf(stderr, "\nillegal layer num!\n");
            break;
    }
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    ggml_graph_tp_processing(cgraph);

    set_decode_chunk_size();
    init_layer_infer_done();

    const int cur_batch_size = cgraph->nodes[1]->ne[1];

    if (is_first_round) {
        // remap_thread_num = core_num * cpus_per_core;
        if (WRITE_LAYER_PARAM) {
            write_layer_param(cgraph);
            printf("model param write done, please restart llama.cpp...\n");
            exit(0);
        }
        remap_thread_num = REMAP_THREAD_NUM; // this value would be mess after first round inference, have no idea yet...
        cur_stage = InferStage::PREFILL;
        cur_switch_stage = SwitchStage::NONE_SWITCH;
        init_mem_block(cgraph);
        realloc_numa_mem_(cgraph);
        init_remap_layer_num_per_batch();
        init_slide_window(cur_batch_size);
//        reset_open_file_limit();
//        open_fd_arr();
        is_first_round = 0;

        // FIXME:
        /** code for throughput test of tp under switching **/
        // remap_model_param_p2d_start();
        // remap_model_param_p2d_end();
        /** code for throughput test of tp under switching **/
    }

    auto new_stage = compute_cur_stage_test(reference_num_, cur_batch_size); // must be PREFILL in first_round

    // printf(" %d ", new_stage);

    if (new_stage == InferStage::SWITCHING_P2D) {
        assert(cur_stage == InferStage::PREFILL || cur_stage == InferStage::SWITCHING_P2D);
        assert(cur_switch_stage == SwitchStage::NONE_SWITCH || cur_switch_stage == SwitchStage::PREFILL_2_DECODE);
        cur_switch_stage = SwitchStage::PREFILL_2_DECODE;
        cur_stage = new_stage;
    }
    if (cur_switch_stage == SwitchStage::PREFILL_2_DECODE) {
        fprintf(stderr, "   switching from PP to TP...\n");
        fflush(stderr);
//        printf("\n switch \n");
// FIXME:
        remap_model_param_p2d_start();
        remap_model_param_p2d_end();
        if (cur_switch_start_layer == total_layers_num - 1) {
            cur_switch_start_layer = 0;
            cur_switch_remain_layer = total_layers_num - 2;
            cur_switch_stage = SwitchStage::NONE_SWITCH;
            cur_stage = InferStage::DECODE;
        }
    }

    if (new_stage == InferStage::SWITCHING_D2P) {
        assert(cur_stage == InferStage::DECODE || cur_stage == InferStage::SWITCHING_D2P);
        assert(cur_switch_stage == SwitchStage::NONE_SWITCH || cur_switch_stage == SwitchStage::DECODE_2_PREFILL);
        cur_switch_stage = SwitchStage::DECODE_2_PREFILL;
        cur_stage = new_stage;
    }
    if (cur_switch_stage == SwitchStage::DECODE_2_PREFILL) {
        fprintf(stderr, "   switching from TP to PP...\n");
        fflush(stderr);
        remap_model_param_d2p_start();
        remap_model_param_d2p_end();
        if (cur_switch_start_layer == total_layers_num - 1) {
            cur_switch_start_layer = 0;
            cur_switch_remain_layer = total_layers_num - 2;
            cur_switch_stage = SwitchStage::NONE_SWITCH;
            cur_stage = InferStage::PREFILL;
        }
    }

    reference_num_ ++;


    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    // total_layers_num good
    auto retval = ggml_graph_compute(cgraph, &cplan);

//    if (cur_switch_stage == SwitchStage::PREFILL_2_DECODE) {
//        remap_model_param_p2d_end();
//        if (cur_switch_start_layer == total_layers_num - 1) {
//            cur_switch_start_layer = 0;
//            cur_switch_remain_layer = total_layers_num - 2;
//            cur_switch_stage = SwitchStage::NONE_SWITCH;
//        }
//    }

//    if (cur_switch_stage == SwitchStage::DECODE_2_PREFILL) {
//        remap_model_param_d2p_end();
//        if (cur_switch_start_layer == total_layers_num - 1) {
//            cur_switch_start_layer = 0;
//            cur_switch_remain_layer = total_layers_num - 2;
//            cur_switch_stage = SwitchStage::NONE_SWITCH;
//        }
//    }
    
    return retval;
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cpu_guid(),
        /* .interface = */ ggml_backend_cpu_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context   = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    return "CPU";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // extra_buffer_op?
    for (auto extra : ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra) {
            auto buf_extra = (ggml::cpu::extra_buffer_type*) extra->context;
            if (buf_extra && buf_extra->supports_op(dev, op)) {
                return true;
            }
        }
    }

    // the other case need host buffer.
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && !ggml_backend_buft_is_host(op->src[i]->buffer->buft)) {
            return false;
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return src1->type == GGML_TYPE_F32 || src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 || (ggml_is_quantized(src0->type) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    static ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_AARCH64
        features.push_back({ "AARCH64_REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
