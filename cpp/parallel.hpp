#pragma once

#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace py = pybind11;

namespace dtp {

// Worker callbacks that contain a long inner loop can consult this flag so a
// Ctrl+C noticed by the coordinating Python thread does not have to wait for
// the current work item to finish.
inline thread_local const std::atomic<bool> *parallel_cancel_flag = nullptr;

inline bool running_in_parallel_worker() {
    return parallel_cancel_flag != nullptr;
}

inline bool interruption_requested() {
    return parallel_cancel_flag &&
           parallel_cancel_flag->load(std::memory_order_relaxed);
}

// Call only from the Python calling thread while its surrounding native code
// has released the GIL. PyErr_CheckSignals raises KeyboardInterrupt for SIGINT.
inline void check_python_signals() {
    py::gil_scoped_acquire acquire;
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
}

// Number of worker threads to use for `work` items, honouring the caller's
// request (<=0 means "all cores") but never exceeding the hardware or the work.
inline std::size_t thread_count(int requested, std::size_t work) {
    const auto available = std::max(1u, std::thread::hardware_concurrency());
    const auto wanted = requested <= 0 ? available : static_cast<unsigned>(requested);
    return std::max<std::size_t>(1, std::min<std::size_t>({wanted, available, work}));
}

inline long current_process_id() {
#ifdef _WIN32
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
#endif
}

// A process-global pool of worker threads that stay parked between batches, so
// the many short native kernels a single plot issues (per-series statistics,
// per-block quantiles/rasters) no longer each create and join a fresh set of
// threads. One batch runs at a time; the submitting (Python) thread blocks
// until it finishes and meanwhile polls for Ctrl+C, exactly as the previous
// per-call implementation did.
class PersistentThreadPool {
public:
    explicit PersistentThreadPool(std::size_t workers) { start(workers); }
    ~PersistentThreadPool() { shutdown(); }

    PersistentThreadPool(const PersistentThreadPool &) = delete;
    PersistentThreadPool &operator=(const PersistentThreadPool &) = delete;

    std::size_t size() const { return workers_.size(); }

    // Run job(i) for i in [0, count) across up to `target` parked workers and
    // block the CALLING thread until the batch completes, rethrowing the first
    // worker exception (or a KeyboardInterrupt raised by the signal poll). Must
    // not be called from a pool worker: nested fan-out takes the serial path so
    // a batch never waits on the very threads running it.
    void run(std::size_t count, std::size_t target,
             std::function<void(std::size_t)> job) {
        if (count == 0) return;
        // Serialise concurrent submitters (e.g. multiple GUI sessions); each
        // batch owns the shared slots below for its whole duration.
        std::lock_guard<std::mutex> submit_guard(submit_mutex_);
        target = std::max<std::size_t>(1, std::min(target, workers_.size()));

        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = &job;
            count_ = count;
            next_.store(0, std::memory_order_relaxed);
            cancelled_.store(false, std::memory_order_relaxed);
            has_failure_.store(false, std::memory_order_relaxed);
            failure_ = nullptr;
            target_ = target;
            active_ = target;
            ++generation_;
        }
        work_ready_.notify_all();

        std::unique_lock<std::mutex> lock(mutex_);
        while (active_ != 0) {
            batch_done_.wait_for(lock, std::chrono::milliseconds(50));
            if (has_failure_.load(std::memory_order_acquire)) {
                // A worker failed; stop the rest and drain before returning.
                cancelled_.store(true, std::memory_order_relaxed);
                continue;
            }
            lock.unlock();
            try {
                check_python_signals();
                lock.lock();
            } catch (...) {
                // Reacquire through the same unique_lock -- mutex_ is not
                // recursive, so a separate lock_guard here would self-deadlock.
                lock.lock();
                if (!failure_) failure_ = std::current_exception();
                has_failure_.store(true, std::memory_order_release);
                cancelled_.store(true, std::memory_order_relaxed);
            }
        }
        job_ = nullptr;
        std::exception_ptr failure = failure_;
        failure_ = nullptr;
        lock.unlock();
        if (failure) std::rethrow_exception(failure);
    }

private:
    void start(std::size_t workers) {
        stop_ = false;
        generation_ = 0;
        active_ = 0;
        workers_.reserve(workers);
        try {
            for (std::size_t index = 0; index < workers; ++index)
                workers_.emplace_back([this, index] { worker_loop(index); });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        work_ready_.notify_all();
        for (auto &worker : workers_)
            if (worker.joinable()) worker.join();
        workers_.clear();
    }

    void worker_loop(std::size_t index) {
        std::size_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            work_ready_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            if (index >= target_) continue;  // not part of this batch
            lock.unlock();

            parallel_cancel_flag = &cancelled_;
            try {
                while (!cancelled_.load(std::memory_order_relaxed)) {
                    const std::size_t i =
                        next_.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count_) break;
                    (*job_)(i);
                }
            } catch (...) {
                std::lock_guard<std::mutex> guard(mutex_);
                if (!failure_) failure_ = std::current_exception();
                has_failure_.store(true, std::memory_order_release);
                cancelled_.store(true, std::memory_order_relaxed);
            }
            parallel_cancel_flag = nullptr;

            lock.lock();
            if (--active_ == 0) batch_done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex submit_mutex_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable batch_done_;
    bool stop_ = false;
    std::size_t generation_ = 0;
    std::size_t target_ = 0;
    std::size_t active_ = 0;
    std::size_t count_ = 0;
    std::atomic<std::size_t> next_{0};
    std::atomic<bool> cancelled_{false};
    std::atomic<bool> has_failure_{false};
    std::exception_ptr failure_;
    const std::function<void(std::size_t)> *job_ = nullptr;
};

// Lazily created, intentionally leaked singleton. Leaking avoids running the
// pool's destructor during interpreter/static teardown, whose ordering against
// other statics is undefined; the parked workers hold no Python state, so the
// OS reclaims them cleanly at process exit. Size for the machine once, then
// apply the requested participation cap independently on every call.
inline PersistentThreadPool &global_pool() {
    static std::mutex init_mutex;
    static PersistentThreadPool *pool = nullptr;
    static long owner_pid = 0;
    std::lock_guard<std::mutex> lock(init_mutex);
    const long pid = current_process_id();
    if (pool != nullptr && owner_pid != pid) {
        // A fork() (e.g. the bootstrap ProcessPoolExecutor) inherited this pool,
        // but the parent's worker threads did not cross the fork. Abandon the
        // inherited object -- its threads are gone and its lock state is
        // indeterminate -- and build a fresh pool for the child process.
        pool = nullptr;
    }
    if (pool == nullptr) {
        pool = new PersistentThreadPool(
            std::max(1u, std::thread::hardware_concurrency()));
        owner_pid = pid;
    }
    return *pool;
}

// Run `function(index)` for index in [0, rows) across the persistent pool,
// pulling indices off a shared atomic counter. Serial when a single worker
// suffices, or when already inside a pool worker so nested fan-out cannot wait
// on the pool that is running it.
template <class Function>
void parallel_rows(std::size_t rows, int requested, Function function) {
    const std::size_t workers = thread_count(requested, rows);
    if (workers <= 1 || running_in_parallel_worker()) {
        for (std::size_t row = 0; row < rows; ++row) {
            if ((row & 63u) == 0) check_python_signals();
            function(row);
        }
        return;
    }
    global_pool().run(rows, workers,
                             std::function<void(std::size_t)>(function));
}

}  // namespace dtp
