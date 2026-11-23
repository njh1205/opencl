#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "hpocl_executor.h"
#include "hpocl_schedule.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#if defined(__linux__)
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#endif

namespace hpocl {
namespace {

size_t environment_worker_count() {
  const char* value = std::getenv("HPOCL_HOST_WORKERS");
  if (!value || !value[0]) return 0;
  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<size_t>::max())
    return 0;
  return static_cast<size_t>(parsed);
}

bool affinity_requested() {
  const char* value = std::getenv("HPOCL_PIN_WORKERS");
  return !value || std::strcmp(value, "0") != 0;
}

#if defined(__linux__)
std::vector<int> allowed_cpus() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return {};
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
  return cpus;
}
#endif

}  // namespace

size_t CpuExecutor::default_worker_count() {
  if (const size_t configured = environment_worker_count()) return configured;
  return std::min<size_t>(4, std::max<size_t>(1, std::thread::hardware_concurrency()));
}

CpuExecutor::CpuExecutor(size_t workers) {
  if (workers == 0) workers = default_worker_count();
#if defined(__linux__)
  if (affinity_requested()) {
    const std::vector<int> cpus = allowed_cpus();
    if (!cpus.empty()) {
      workers = std::min(workers, cpus.size());
      worker_cpus_.assign(cpus.begin(), cpus.begin() + workers);
      affinity_enabled_ = true;
    }
  }
#endif
  workers_.reserve(workers);
  try {
    for (size_t i = 0; i < workers; ++i)
      workers_.emplace_back(&CpuExecutor::worker_loop, this, i);
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    work_ready_.notify_all();
    for (std::thread& worker : workers_)
      if (worker.joinable()) worker.join();
    throw;
  }
}

CpuExecutor::~CpuExecutor() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  work_ready_.notify_all();
  for (std::thread& worker : workers_)
    if (worker.joinable()) worker.join();
}

void CpuExecutor::parallel_for(size_t count,
                               const std::function<void(size_t)>& task) {
  if (count == 0) return;
  if (!task) throw std::invalid_argument("CpuExecutor requires a task");

  std::unique_lock<std::mutex> lock(mutex_);
  if (task_) throw std::logic_error("CpuExecutor does not support nested parallel_for");

  task_ = task;
  task_count_ = count;
  next_index_ = 0;
  finished_workers_ = 0;
  failure_ = nullptr;
  ++generation_;
  work_ready_.notify_all();
  work_done_.wait(lock, [this] { return finished_workers_ == workers_.size(); });

  std::exception_ptr failure = failure_;
  task_ = {};
  task_count_ = 0;
  if (failure) std::rethrow_exception(failure);
}

void schedule_host_batch(CpuExecutor& executor, const HostScheduleJob* jobs,
                         size_t count, PipeOut* outputs) {
  if (count == 0) return;
  if (!jobs || !outputs)
    throw std::invalid_argument("schedule_host_batch requires jobs and outputs");
  for (size_t i = 0; i < count; ++i)
    if (!jobs[i].H)
      throw std::invalid_argument("schedule_host_batch received a null channel pointer");

  executor.parallel_for(count, [jobs, outputs](size_t i) {
    schedule(jobs[i].H, jobs[i].config, outputs[i]);
  });
}

void CpuExecutor::worker_loop(size_t worker) {
#if defined(__linux__)
  if (affinity_enabled_) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(worker_cpus_[worker], &mask);
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    if (error != 0) {
      std::fprintf(stderr, "[hpocl] worker %zu affinity to CPU %d failed: %s\n",
                   worker, worker_cpus_[worker], std::strerror(error));
    }
  }
#endif

  size_t observed_generation = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    work_ready_.wait(lock, [this, &observed_generation] {
      return stopping_ || generation_ != observed_generation;
    });
    if (stopping_) return;
    observed_generation = generation_;

    while (next_index_ < task_count_) {
      const size_t index = next_index_++;
      lock.unlock();
      try {
        task_(index);
      } catch (...) {
        lock.lock();
        if (!failure_) failure_ = std::current_exception();
        next_index_ = task_count_;
        continue;
      }
      lock.lock();
    }

    ++finished_workers_;
    if (finished_workers_ == workers_.size()) work_done_.notify_one();
  }
}

}  // namespace hpocl
