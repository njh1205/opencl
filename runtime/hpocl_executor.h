#pragma once
//------------------------------------------------------------------------------
// Bounded host-side executor for independent control-plane jobs. ZU49DR has
// four Linux-capable Cortex-A53 application cores; its two Cortex-R5 cores do
// not run this Linux/OpenCL host runtime.
//------------------------------------------------------------------------------
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace hpocl {

class CpuExecutor {
 public:
  // workers == 0 selects up to four workers. HPOCL_HOST_WORKERS overrides that
  // default. Linux workers are pinned to distinct CPUs by default; set
  // HPOCL_PIN_WORKERS=0 to retain scheduler-managed placement.
  explicit CpuExecutor(size_t workers = 0);
  ~CpuExecutor();

  CpuExecutor(const CpuExecutor&) = delete;
  CpuExecutor& operator=(const CpuExecutor&) = delete;

  size_t worker_count() const { return workers_.size(); }
  bool affinity_enabled() const { return affinity_enabled_; }
  int worker_cpu(size_t worker) const {
    return worker < worker_cpus_.size() ? worker_cpus_[worker] : -1;
  }
  static size_t default_worker_count();

  // Execute task(i), i in [0, count), exactly once. Calls are serialized per
  // executor; task exceptions stop new work, wait for active workers, then
  // rethrow on the caller thread.
  void parallel_for(size_t count, const std::function<void(size_t)>& task);

 private:
  void worker_loop(size_t worker);

  std::vector<std::thread> workers_;
  std::vector<int> worker_cpus_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  std::function<void(size_t)> task_;
  std::exception_ptr failure_;
  size_t task_count_ = 0;
  size_t next_index_ = 0;
  size_t finished_workers_ = 0;
  size_t generation_ = 0;
  bool stopping_ = false;
  bool affinity_enabled_ = false;
};

}  // namespace hpocl
