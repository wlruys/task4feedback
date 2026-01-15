// Run:
//   ./replay_starpu dag.pkl
// Notes:
//  - Pickle files are generated from either
// static_jacobi_eval.py or
// static_jacobi_eval_fix_compute.py

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <starpu.h>
#include <starpu_stdlib.h>

#ifdef STARPU_USE_CUDA
#include <cuda_runtime.h>
#include <starpu_cuda.h>
#else
#error "StarPU was built without CUDA support (STARPU_USE_CUDA not defined)."
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct BlockInfo {
  int location = 0;              // CUDA device id (e.g., 0..3)
  size_t size_bytes = 0;         // bytes
  void *host_ptr = nullptr;      // host allocation backing the handle
  starpu_data_handle_t handle{}; // StarPU handle
};

struct TaskInfo {
  std::string id;
  starpu_tag_t tag = 0;
  std::vector<std::string> deps;
  std::vector<std::string> read;
  std::vector<std::string> write;
  int mapped_location = -1; // CUDA device id
  float duration_us = 0.0f;
};

// ----------------------- CUDA "dummy compute" codelet -----------------------

// Spin for ~cycles using clock64 (no memory access => safe for R/W/RW buffers)
__global__ void spin_kernel(unsigned long long cycles) {
  unsigned long long start = clock64();
  while ((clock64() - start) < cycles) {
    // busy wait
  }
}

static inline unsigned long long us_to_cycles(double sleep_us, const cudaDeviceProp *prop) {
  // prop->clockRate is in kHz
  // cycles = (clockRate * 1000 cycles/s) * (sleep_us * 1e-6 s)
  //        = clockRate * sleep_us / 1000
  double cycles_d = (double)prop->clockRate * sleep_us / 1000.0;
  if (cycles_d < 0.0)
    cycles_d = 0.0;
  return (unsigned long long)cycles_d;
}

static void dummy_cuda(void *buffers[], void *cl_arg) {
  (void)buffers; // we don't need to touch data for a compute-only emulation

  float sleep_us_f = 0.0f;
  if (cl_arg) {
    starpu_codelet_unpack_args(cl_arg, &sleep_us_f);
  }
  double sleep_us = (sleep_us_f > 0.0f) ? (double)sleep_us_f : 0.0;

  // Identify which StarPU worker (CUDA worker thread) we are running on
  int workerid = starpu_worker_get_id();

  // Get GPU properties for this worker's device
  const cudaDeviceProp *prop = starpu_cuda_get_device_properties(workerid);
  if (!prop) {
    // Should not happen on a CUDA worker, but fail loudly if it does.
    fprintf(stderr, "dummy_cuda: starpu_cuda_get_device_properties returned NULL\n");
    return;
  }

  unsigned long long cycles = us_to_cycles(sleep_us, prop);

  // StarPU-provided stream for this worker
  cudaStream_t stream = starpu_cuda_get_local_stream();

  // Run the busy-wait kernel on the GPU
  spin_kernel<<<1, 1, 0, stream>>>(cycles);

  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    STARPU_CUDA_REPORT_ERROR(st);
  }

  // Synchronous completion (like your Parla-style synchronize)
  cudaStreamSynchronize(stream);
}

static struct starpu_codelet g_cl;

// ----------------------- Python/pickle helpers -----------------------

static void die_py(const char *msg) {
  std::cerr << "Python error: " << msg << "\n";
  PyErr_Print();
  std::exit(1);
}

static std::string pyobj_to_string(PyObject *obj) {
  PyObject *s = PyObject_Str(obj); // new ref
  if (!s)
    die_py("PyObject_Str failed");
  const char *c = PyUnicode_AsUTF8(s);
  if (!c)
    die_py("PyUnicode_AsUTF8 failed");
  std::string out(c);
  Py_DECREF(s);
  return out;
}

static std::vector<std::string> pyseq_to_strings(PyObject *seq) {
  std::vector<std::string> out;
  if (!seq || seq == Py_None)
    return out;

  PyObject *fast = PySequence_Fast(seq, "expected a sequence");
  if (!fast) {
    PyErr_Clear();
    return out;
  }

  Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
  out.reserve(static_cast<size_t>(n));
  PyObject **items = PySequence_Fast_ITEMS(fast);
  for (Py_ssize_t i = 0; i < n; i++) {
    out.push_back(pyobj_to_string(items[i]));
  }

  Py_DECREF(fast);
  return out;
}

static void load_pickle(const char *path, std::vector<TaskInfo> &tasks,
                        std::unordered_map<std::string, BlockInfo> &blocks,
                        std::unordered_map<unsigned, unsigned> &cell_locations, long &simtime)

{
  Py_Initialize();
  if (!Py_IsInitialized()) {
    std::cerr << "Failed to initialize Python.\n";
    std::exit(1);
  }

  PyObject *pickle_mod = PyImport_ImportModule("pickle");
  if (!pickle_mod)
    die_py("import pickle failed");

  PyObject *builtins = PyImport_ImportModule("builtins");
  if (!builtins)
    die_py("import builtins failed");

  PyObject *open_func = PyObject_GetAttrString(builtins, "open");
  if (!open_func)
    die_py("getattr(builtins, open) failed");

  PyObject *file_obj = PyObject_CallFunction(open_func, "ss", path, "rb");
  if (!file_obj)
    die_py("open(pickle_path, 'rb') failed");

  PyObject *load_func = PyObject_GetAttrString(pickle_mod, "load");
  if (!load_func)
    die_py("getattr(pickle, load) failed");

  PyObject *root = PyObject_CallFunctionObjArgs(load_func, file_obj, nullptr);
  if (!root)
    die_py("pickle.load failed");

  if (!PyDict_Check(root)) {
    std::cerr << "Top-level pickle object must be a dict.\n";
    std::exit(1);
  }

  PyObject *task_dict = PyDict_GetItemString(root, "task");           // borrowed
  PyObject *block_dict = PyDict_GetItemString(root, "data");          // borrowed
  PyObject *simtime_int = PyDict_GetItemString(root, "time");         // borrowed
  PyObject *cell_dict = PyDict_GetItemString(root, "cell_locations"); // borrowed

  if (!task_dict || !PyDict_Check(task_dict)) {
    std::cerr << "pickle['task'] must be a dict\n";
    std::exit(1);
  }
  if (!block_dict || !PyDict_Check(block_dict)) {
    std::cerr << "pickle['data'] must be a dict\n";
    std::exit(1);
  }
  if (!simtime_int || !PyLong_Check(simtime_int)) {
    std::cerr << "pickle['time'] must be an int\n";
    std::exit(1);
  }
  if (!cell_dict || !PyDict_Check(cell_dict)) {
    std::cerr << "pickle['cell_locations'] must be a dict\n";
    std::exit(1);
  }
  simtime = PyLong_AsLong(simtime_int);

  // Parse blocks
  {
    Py_ssize_t pos = 0;
    PyObject *key = nullptr;
    PyObject *val = nullptr;
    while (PyDict_Next(block_dict, &pos, &key, &val)) {
      std::string bid = pyobj_to_string(key);

      if (!PyDict_Check(val)) {
        std::cerr << "block[" << bid << "] must be a dict\n";
        std::exit(1);
      }
      PyObject *loc_obj = PyDict_GetItemString(val, "location");
      PyObject *size_obj = PyDict_GetItemString(val, "size");
      if (!loc_obj || !size_obj) {
        std::cerr << "block[" << bid << "] missing location/size\n";
        std::exit(1);
      }

      long loc = PyLong_AsLong(loc_obj);
      if (PyErr_Occurred())
        die_py("block.location not an int");
      unsigned long long sz = PyLong_AsUnsignedLongLong(size_obj);
      if (PyErr_Occurred())
        die_py("block.size not an int");

      BlockInfo b;
      b.location = static_cast<int>(loc);
      b.size_bytes = static_cast<size_t>(sz);
      blocks.emplace(std::move(bid), std::move(b));
    }
  }

  // Parse tasks
  {
    Py_ssize_t pos = 0;
    PyObject *key = nullptr;
    PyObject *val = nullptr;
    while (PyDict_Next(task_dict, &pos, &key, &val)) {
      std::string tid = pyobj_to_string(key);

      if (!PyDict_Check(val)) {
        std::cerr << "task[" << tid << "] must be a dict\n";
        std::exit(1);
      }

      PyObject *deps_obj = PyDict_GetItemString(val, "dependencies");
      PyObject *read_obj = PyDict_GetItemString(val, "read");
      PyObject *write_obj = PyDict_GetItemString(val, "write");
      PyObject *loc_obj = PyDict_GetItemString(val, "mapped_location");
      PyObject *dur_obj = PyDict_GetItemString(val, "duration");

      TaskInfo t;
      t.id = std::move(tid);
      t.deps = pyseq_to_strings(deps_obj);
      t.read = pyseq_to_strings(read_obj);
      t.write = pyseq_to_strings(write_obj);

      if (loc_obj && loc_obj != Py_None) {
        t.mapped_location = (int)PyLong_AsLong(loc_obj);
        if (PyErr_Occurred())
          die_py("task.mapped_location not int");
      }

      if (dur_obj && dur_obj != Py_None) {
        t.duration_us = (float)PyFloat_AsDouble(dur_obj);
        if (PyErr_Occurred())
          die_py("task.duration not float");
      }

      tasks.emplace_back(std::move(t));
    }
  }

  // Parse cell_locations
  {
    if (cell_dict && PyDict_Check(cell_dict)) {
      Py_ssize_t pos = 0;
      PyObject *key = nullptr;
      PyObject *val = nullptr;
      while (PyDict_Next(cell_dict, &pos, &key, &val)) {
        unsigned cell_id = (unsigned)PyLong_AsUnsignedLong(key);
        if (PyErr_Occurred())
          die_py("cell_locations key not unsigned int");
        unsigned loc = (unsigned)PyLong_AsUnsignedLong(val);
        if (PyErr_Occurred())
          die_py("cell_locations value not unsigned int");
        cell_locations.emplace(cell_id, loc);
        // printf("Cell %u -> location %u\n", cell_id, loc);
      }
    }
  }

  Py_DECREF(root);
  Py_DECREF(load_func);
  Py_DECREF(file_obj);
  Py_DECREF(open_func);
  Py_DECREF(builtins);
  Py_DECREF(pickle_mod);

  Py_FinalizeEx();
}

// ----------------------- CLI helpers -----------------------

static void usage(const char *argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " <dag.pkl> [--task-delay <float>] [--sched <name>]\n\n"
      << "Options:\n"
      << "  --task-delay <float> Extra per-task overhead to subtract (microseconds, default 5)\n"
      << "  --sched <name>       StarPU scheduler policy name (e.g., eager, dmda, heft)\n";
}

// ----------------------- Main -----------------------

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }

  const char *pickle_path = argv[1];
  float task_delay = 0.0f;
  std::string sched_name;

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--task-delay" && i + 1 < argc) {
      task_delay = std::stof(argv[++i]);
    } else if (a == "--sched" && i + 1 < argc) {
      sched_name = argv[++i];
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  // 1) Load pickle
  std::vector<TaskInfo> tasks;
  std::unordered_map<std::string, BlockInfo> blocks;
  std::unordered_map<unsigned, unsigned> cell_locations;
  long simtime = 0;
  load_pickle(pickle_path, tasks, blocks, cell_locations, simtime);

  std::cout << "Loaded " << tasks.size() << " tasks and " << blocks.size() << " blocks\n";
  std::cout << "Simtime: " << simtime << " microseconds\n";

  // 2) Assign unique tags
  std::unordered_map<std::string, starpu_tag_t> task_tag;
  task_tag.reserve(tasks.size());
  starpu_tag_t next_tag = 1;
  for (auto &t : tasks) {
    t.tag = next_tag++;
    task_tag.emplace(t.id, t.tag);
  }

  // 3) Init StarPU
  struct starpu_conf conf;
  starpu_conf_init(&conf);

  char *sched_dup = nullptr;
  if (!sched_name.empty()) {
    sched_dup = strdup(sched_name.c_str());
    conf.sched_policy_name = sched_dup;
  }

  int ret = starpu_init(&conf);
  if (ret != 0) {
    std::cerr << "starpu_init failed: " << ret << "\n";
    return 1;
  }

  unsigned ngpus = starpu_cuda_worker_get_count();
  if (ngpus == 0) {
    std::cerr << "No StarPU CUDA workers available.\n";
    starpu_shutdown();
    return 1;
  }

  // Setup global codelet (CUDA-only)
  std::memset(&g_cl, 0, sizeof(g_cl));
  g_cl.where = STARPU_CUDA;
  g_cl.cuda_funcs[0] = dummy_cuda;
  g_cl.nbuffers = STARPU_VARIABLE_NBUFFERS;
  g_cl.name = "gpu_spin";

  // 4) Build CUDA devid -> StarPU CUDA RAM node mapping
  std::unordered_map<int, unsigned> cuda_devid_to_node;
  {
    unsigned n_cuda_nodes = starpu_memory_nodes_get_count_by_kind(STARPU_CUDA_RAM);
    std::vector<unsigned> nodes(n_cuda_nodes);
    unsigned got = starpu_memory_node_get_ids_by_type(
        STARPU_CUDA_RAM, nodes.empty() ? nullptr : nodes.data(), (unsigned)nodes.size());
    nodes.resize(got);

    for (unsigned node : nodes) {
      int devid = starpu_memory_node_get_devid(node);
      cuda_devid_to_node[devid] = node;
    }

    std::cout << "CUDA devid -> StarPU node mapping:\n";
    for (const auto &kv : cuda_devid_to_node) {
      std::cout << "  CUDA " << kv.first << " -> node " << kv.second << "\n";
    }
  }

  // Also build CUDA devid -> StarPU workerid mapping (robust pinning)
  std::unordered_map<int, int> cuda_devid_to_worker;
  {
    for (unsigned i = 0; i < ngpus; i++) {
      int wid = starpu_worker_get_by_type(STARPU_CUDA_WORKER, i);
      int devid = starpu_worker_get_devid(wid);
      cuda_devid_to_worker[devid] = wid;
    }
  }

  // 5) Register blocks as GPU-only (Lazy Allocation)
  //    We remove starpu_malloc and host pointers entirely.
  for (auto &kv : blocks) {
    BlockInfo &b = kv.second;

    if (b.size_bytes == 0)
      b.size_bytes = 1;

    // CHANGE 1: Register with home_node = -1 and ptr = 0.
    // This tells StarPU: "Data is not in RAM, allocate it on the worker node when needed."
    starpu_variable_data_register(&b.handle, -1, 0, b.size_bytes);

    // CHANGE 2: Handle "Input" blocks.
    // With lazy allocation, the data is "invalid" everywhere. If the first task
    // READS this block, StarPU will hang/error waiting for data.
    // We assume 'b.location' is the preferred GPU. We force a dummy initialization there.
    auto it = cuda_devid_to_node.find(b.location);
    if (it != cuda_devid_to_node.end()) {
      // We use a temporary write-only handle access to force allocation on the GPU
      // without actually copying anything (since source is null).
      // Note: In a real app, you would copy input data here.
      // For replay, we just want the memory allocated and marked "Valid".

      // Explicitly request allocation on the target node
      starpu_data_acquire_on_node(b.handle, it->second, STARPU_W);
      starpu_data_release_on_node(b.handle, it->second);
    }
  }

  // 6) Declare explicit dependencies using tags
  for (const auto &t : tasks) {
    if (t.deps.empty())
      continue;
    std::vector<starpu_tag_t> deps_tags;
    deps_tags.reserve(t.deps.size());
    for (const auto &dep_id : t.deps) {
      auto it = task_tag.find(dep_id);
      if (it == task_tag.end()) {
        std::cerr << "Error: task '" << t.id << "' depends on unknown task '" << dep_id << "'\n";
        starpu_shutdown();
        return 1;
      }
      deps_tags.push_back(it->second);
    }
    starpu_tag_declare_deps_array(t.tag, (unsigned)deps_tags.size(), deps_tags.data());
  }
  starpu_pause();
  starpu_worker_wait_for_initialisation();

  // Warm-up: Submit first 64 tasks to "prime" the system
  for (const auto &t : tasks) {
    if (t.tag > 64 * 32) {
      break;
    }
    float sleep_us = std::max(0.0f, t.duration_us - task_delay);

    std::unordered_set<std::string> rset(t.read.begin(), t.read.end());
    std::unordered_set<std::string> wset(t.write.begin(), t.write.end());

    std::vector<std::string> all;
    all.reserve(rset.size() + wset.size());
    for (const auto &x : rset)
      all.push_back(x);
    for (const auto &x : wset)
      if (!rset.count(x))
        all.push_back(x);
    std::sort(all.begin(), all.end());

    std::vector<starpu_data_descr> descr;
    descr.reserve(all.size());

    for (const auto &bid : all) {
      auto bit = blocks.find(bid);
      if (bit == blocks.end()) {
        std::cerr << "Error: task '" << t.id << "' references unknown block '" << bid << "'\n";
        starpu_shutdown();
        return 1;
      }

      bool in_r = rset.count(bid) != 0;
      bool in_w = wset.count(bid) != 0;

      enum starpu_data_access_mode mode = (in_r && in_w) ? STARPU_RW : (in_w ? STARPU_W : STARPU_R);

      starpu_data_descr d;
      d.handle = bit->second.handle;
      d.mode = mode;
      descr.push_back(d);
    }

    int forced_worker = -1;
    // auto itw = cuda_devid_to_worker.find(cell_locations.at(std::stoul(t.id)%64));
    // if (itw != cuda_devid_to_worker.end()) forced_worker = itw->second;
    // else {
    //     std::cerr << "Error: task '" << t.id
    //               << "' has no cell location mapping\n";
    //     starpu_shutdown();
    //     return 1;
    // }

    auto itw = cuda_devid_to_worker.find(t.mapped_location);
    if (itw != cuda_devid_to_worker.end())
      forced_worker = itw->second;
    else {
      std::cerr << "Error: task '" << t.id << "' has no mapped location\n";
      starpu_shutdown();
      return 1;
    }

    ret = starpu_task_insert(&g_cl, STARPU_DATA_MODE_ARRAY, descr.data(), (int)descr.size(),
                             STARPU_VALUE, &sleep_us, sizeof(sleep_us), STARPU_TAG,
                             t.tag + 10000000, STARPU_EXECUTE_ON_WORKER, forced_worker,
                             STARPU_PRIORITY, 10000000 - t.tag, 0);

    if (ret != 0) {
      std::cerr << "starpu_task_insert failed for task '" << t.id << "': " << ret << "\n";
      starpu_shutdown();
      return 1;
    }
  }
  starpu_resume();
  starpu_task_wait_for_all();

  for (const auto &t : tasks) {
    if (t.tag > 64) {
      break;
    }
    float sleep_us = std::max(0.0f, t.duration_us - task_delay);

    std::unordered_set<std::string> rset(t.read.begin(), t.read.end());
    std::unordered_set<std::string> wset(t.write.begin(), t.write.end());

    std::vector<std::string> all;
    all.reserve(rset.size() + wset.size());
    for (const auto &x : rset)
      all.push_back(x);
    for (const auto &x : wset)
      if (!rset.count(x))
        all.push_back(x);
    std::sort(all.begin(), all.end());

    std::vector<starpu_data_descr> descr;
    descr.reserve(all.size());

    for (const auto &bid : all) {
      auto bit = blocks.find(bid);
      if (bit == blocks.end()) {
        std::cerr << "Error: task '" << t.id << "' references unknown block '" << bid << "'\n";
        starpu_shutdown();
        return 1;
      }

      bool in_r = rset.count(bid) != 0;
      bool in_w = wset.count(bid) != 0;

      enum starpu_data_access_mode mode = (in_r && in_w) ? STARPU_RW : (in_w ? STARPU_W : STARPU_R);

      starpu_data_descr d;
      d.handle = bit->second.handle;
      d.mode = mode;
      descr.push_back(d);
    }

    int forced_worker = -1;
    // auto itw = cuda_devid_to_worker.find(cell_locations.at(std::stoul(t.id)%64));
    // if (itw != cuda_devid_to_worker.end()) forced_worker = itw->second;
    // else {
    //     std::cerr << "Error: task '" << t.id
    //               << "' has no cell location mapping\n";
    //     starpu_shutdown();
    //     return 1;
    // }

    auto itw = cuda_devid_to_worker.find(t.mapped_location);
    if (itw != cuda_devid_to_worker.end())
      forced_worker = itw->second;
    else {
      std::cerr << "Error: task '" << t.id << "' has no mapped location\n";
      starpu_shutdown();
      return 1;
    }

    ret = starpu_task_insert(&g_cl, STARPU_DATA_MODE_ARRAY, descr.data(), (int)descr.size(),
                             STARPU_VALUE, &sleep_us, sizeof(sleep_us), STARPU_TAG,
                             t.tag + 10000000, STARPU_EXECUTE_ON_WORKER, forced_worker,
                             STARPU_PRIORITY, 10000000 - t.tag, 0);

    if (ret != 0) {
      std::cerr << "starpu_task_insert failed for task '" << t.id << "': " << ret << "\n";
      starpu_shutdown();
      return 1;
    }
  }
  starpu_resume();
  starpu_task_wait_for_all();
  // for (const auto& t : tasks) {
  //      if (t.tag > 128){
  //          break;
  //      }
  //     float sleep_us = std::max(0.0f, t.duration_us - task_delay);

  //     std::unordered_set<std::string> rset(t.read.begin(), t.read.end());
  //     std::unordered_set<std::string> wset(t.write.begin(), t.write.end());

  //     std::vector<std::string> all;
  //     all.reserve(rset.size() + wset.size());
  //     for (const auto& x : rset) all.push_back(x);
  //     for (const auto& x : wset) if (!rset.count(x)) all.push_back(x);
  //     std::sort(all.begin(), all.end());

  //     std::vector<starpu_data_descr> descr;
  //     descr.reserve(all.size());

  //     for (const auto& bid : all) {
  //         auto bit = blocks.find(bid);
  //         if (bit == blocks.end()) {
  //             std::cerr << "Error: task '" << t.id
  //                       << "' references unknown block '" << bid << "'\n";
  //             starpu_shutdown();
  //             return 1;
  //         }

  //         bool in_r = rset.count(bid) != 0;
  //         bool in_w = wset.count(bid) != 0;

  //         enum starpu_data_access_mode mode =
  //             (in_r && in_w) ? STARPU_RW : (in_w ? STARPU_W : STARPU_R);

  //         starpu_data_descr d;
  //         d.handle = bit->second.handle;
  //         d.mode = mode;
  //         descr.push_back(d);
  //     }

  //     int forced_worker = -1;
  //     auto itw = cuda_devid_to_worker.find(cell_locations.at(std::stoul(t.id)%64));
  //     if (itw != cuda_devid_to_worker.end()) forced_worker = itw->second;
  //     else {
  //         std::cerr << "Error: task '" << t.id
  //                   << "' has no cell location mapping\n";
  //         starpu_shutdown();
  //         return 1;
  //     }

  //     // auto itw = cuda_devid_to_worker.find(t.mapped_location);
  //     // if (itw != cuda_devid_to_worker.end()) forced_worker = itw->second;
  //     // else {
  //     //     std::cerr << "Error: task '" << t.id
  //     //               << "' has no mapped location\n";
  //     //     starpu_shutdown();
  //     //     return 1;
  //     // }

  //     ret = starpu_task_insert(
  //         &g_cl,
  //         STARPU_DATA_MODE_ARRAY, descr.data(), (int)descr.size(),
  //         STARPU_VALUE, &sleep_us, sizeof(sleep_us),
  //         STARPU_TAG, t.tag+10000000,
  //         STARPU_EXECUTE_ON_WORKER, forced_worker,
  //         STARPU_PRIORITY, 10000000 - t.tag,
  //         0);

  //     if (ret != 0) {
  //         std::cerr << "starpu_task_insert failed for task '" << t.id << "': " << ret << "\n";
  //         starpu_shutdown();
  //         return 1;
  //     }
  // }
  // starpu_resume();
  // starpu_task_wait_for_all();

  // 7) Submit tasks
  for (const auto &t : tasks) {
    float sleep_us = std::max(0.0f, t.duration_us - task_delay);

    std::unordered_set<std::string> rset(t.read.begin(), t.read.end());
    std::unordered_set<std::string> wset(t.write.begin(), t.write.end());

    std::vector<std::string> all;
    all.reserve(rset.size() + wset.size());
    for (const auto &x : rset)
      all.push_back(x);
    for (const auto &x : wset)
      if (!rset.count(x))
        all.push_back(x);
    std::sort(all.begin(), all.end());

    std::vector<starpu_data_descr> descr;
    descr.reserve(all.size());

    for (const auto &bid : all) {
      auto bit = blocks.find(bid);
      if (bit == blocks.end()) {
        std::cerr << "Error: task '" << t.id << "' references unknown block '" << bid << "'\n";
        starpu_shutdown();
        return 1;
      }

      bool in_r = rset.count(bid) != 0;
      bool in_w = wset.count(bid) != 0;

      enum starpu_data_access_mode mode = (in_r && in_w) ? STARPU_RW : (in_w ? STARPU_W : STARPU_R);

      starpu_data_descr d;
      d.handle = bit->second.handle;
      d.mode = mode;
      descr.push_back(d);
    }

    int forced_worker = -1;
    if (t.mapped_location >= 0) {
      auto itw = cuda_devid_to_worker.find(t.mapped_location);
      if (itw != cuda_devid_to_worker.end())
        forced_worker = itw->second;
    }

    if (forced_worker >= 0) {
      ret = starpu_task_insert(&g_cl, STARPU_DATA_MODE_ARRAY, descr.data(), (int)descr.size(),
                               STARPU_VALUE, &sleep_us, sizeof(sleep_us), STARPU_TAG, t.tag,
                               STARPU_EXECUTE_ON_WORKER, forced_worker, STARPU_PRIORITY,
                               10000000 - t.tag, 0);
    } else {
      ret = starpu_task_insert(&g_cl, STARPU_DATA_MODE_ARRAY, descr.data(), (int)descr.size(),
                               STARPU_VALUE, &sleep_us, sizeof(sleep_us), STARPU_TAG, t.tag,
                               STARPU_PRIORITY, 10000000 - t.tag, 0);
    }

    if (ret != 0) {
      std::cerr << "starpu_task_insert failed for task '" << t.id << "': " << ret << "\n";
      starpu_shutdown();
      return 1;
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  starpu_resume();
  starpu_task_wait_for_all();
  auto t1 = std::chrono::steady_clock::now();

  std::chrono::duration<double> dt = t1 - t0;
  std::cout << "Runtime : " << dt.count() << " s\n";

  // 8) Cleanup
  for (auto &kv : blocks) {
    BlockInfo &b = kv.second;

    // CHANGE 3: Unregister without forcing a copy back to RAM
    starpu_data_unregister_no_coherency(b.handle);

    // Removed: starpu_free_noflag(b.host_ptr, ...)
    // (Since we never allocated it)
  }

  starpu_shutdown();
  if (sched_dup)
    free(sched_dup);
  return 0;
}
