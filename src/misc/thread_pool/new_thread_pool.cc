#include <mutex>
#include <deque>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <functional>
#include <condition_variable>

/**
 * Task can be rather simple
 * we only need data and an executor
 * how to make task data type agnostic?
 * It is simple, we use lambda to capture all the data (maybe, by value)
 * and we only place a fixed parameter for the function
 * there is no need for Task
*/

/**
 * A typical thread pool definition with mutex and cv
 * and task queue. Try to make this thread pool task queue as general as
 * possible
*/

template <bool Ordered = false>
class ThreadPool {
    const int _num_worker;
    const int _queue_size;         // there could be no more pending works than this number
    std::atomic<int> _task_cnt;
    std::deque<std::function<void(int, int)>> task_queue;

    bool _is_closed;
    std::mutex _qmtx;
    std::condition_variable _qcv;

    std::vector<std::thread> _tpool;
public:
    ThreadPool(int num_workers, int queue_size): 
        _num_worker(num_workers), _queue_size(queue_size), 
        _task_cnt(0), _is_closed(false)
    {
        // create parallel workers here
        for (int i = 0; i < num_workers; i++) {
            _tpool.emplace_back(&ThreadPool::process_task, this, i);
        }
    }

    ~ThreadPool() {
        _is_closed = true;
        _qcv.notify_all();
        printf("[DECONSTRUCT] Starts.\n");
        for (auto& t: _tpool) {
            t.join();
        }
        printf("[DECONSTRUCT] Completed.\n");
    }

    void shutdown() {
        _is_closed = true;
        _qcv.notify_all();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // move-able
    ThreadPool(ThreadPool&& src): _num_worker(src._num_worker), _queue_size(src._queue_size) 
    {
        // when to use exchange? maybe pointer?
        this->_task_cnt = std::__exchange(src._task_cnt, 0);
        this->task_queue = std::__exchange(src.task_queue, {});
        this->_qmtx = std::move(src._qmtx);      // what will be left behind?
        this->_qcv  = std::move(src._qcv);
        this->_tpool = std::__exchange(src._tpool, {});
    }

    // I wonder how this works --- move assign with constant member
    // ThreadPool& operator=(ThreadPool&& src): _num_worker(src._num_worker), _queue_size(src._queue_size) 
    // {
    //     // when to use exchange? maybe pointer?
    //     this->_task_cnt = std::__exchange(src._task_cnt, 0);
    //     this->task_queue = std::__exchange(src.task_queue, {});
    //     this->_qmtx = std::move(src._qmtx);      // what will be left behind?
    //     this->_qcv  = std::move(src._qcv);
    //     this->_tpool = std::__exchange(src._tpool, {});
    // }

    void enqueue_task(std::function<void(int, int)> task) {
        // task_id is the ID to be executed
        std::lock_guard<std::mutex> lock(_qmtx);
        task_queue.emplace_back(std::move(task));
        if (task_queue.size() > _queue_size) {
            task_queue.pop_front();
        }
        if constexpr (Ordered) {
            _qcv.notify_one();          // only the first one in the task queue
        } else {
            _qcv.notify_all();
        }
    }   

    void process_task(int worker_id) {
        // thread will call this function
        printf("[START] Worker %d starts execution loop...\n", worker_id);
        while (!this->_is_closed) {
            std::unique_lock<std::mutex> lock(_qmtx);
            // condition variable wait on this predicate to be true
            // when notified, otherwise the CPU time-fragment? will be released
            _qcv.wait(lock, [this](){
                return !this->task_queue.empty() || this->_is_closed; 
            });
            if (_is_closed) break;
            std::function<void(int, int)> work = task_queue.front();
            task_queue.pop_front();
            lock.unlock();
            work(_task_cnt++, worker_id);
        };
        printf("[END] Worker %d ends execution loop.\n", worker_id);
    }
};

int main() {
    ThreadPool pool(8, 512);

    for (int i = 0; i < 512; i++) {
        if (i & 1) {
            auto task = [i](int task_id, int worker_id) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                printf("[ODD]\t[task %4d \t worker: %2d] \tOdd task %d.\n", task_id, worker_id, i);
            };
            pool.enqueue_task(task);
        } else {
            auto task = [i](int task_id, int worker_id) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                printf("[EVEN]\t[task %4d \t worker: %2d] \tEven task %d.\n", task_id, worker_id, i);
            };
            pool.enqueue_task(task);
        }
    }
    printf("[WORK ENQUEUED] All works are pushed into the working queue.\n");
    printf("[WAIT] Main thread sleep for 500ms to wait for task completion.\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(6000));
    pool.shutdown();
    printf("[EXIT] Main thread exiting...\n");
    return 0;
}