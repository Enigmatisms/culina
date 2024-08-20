#include "thread_pool.hpp"

template <typename DataType>
ThreadPool<DataType>::ThreadPool(size_t num_workers): invalid(false) {
    thread_pool.reserve(num_workers);
    for (size_t i = 0; i < num_workers; i++) {
        thread_pool.emplace_back(std::bind(&ThreadPool<DataType>::executeWork, this));
    }
}

template <typename DataType>
ThreadPool<DataType>::~ThreadPool() {
    for (size_t i = 0; i < thread_pool.size(); i++) {
        thread_pool[i].join();
    }
}

template <typename DataType>
void ThreadPool<DataType>::enQueue(std::function<void(DataType&)> executor, DataType&& data) {
    std::unique_lock<std::mutex> lock(mtx);
    if (invalid) return;
    executors.push_front(executor);
    resources.push_front(data);
    // uniquely occupy the job, since a pop operation is going to be used
    cv.notify_one();
}

template <typename DataType>
void ThreadPool<DataType>::shut_down() {
    std::unique_lock<std::mutex> lock(mtx);
    invalid = true;
    printf("Shut down request sent......\n");
    cv.notify_all();        // let all the threads exit
}

template <typename DataType>
void ThreadPool<DataType>::executeWork() {
    while (true) {
        std::function<void(DataType&)> executor;
        DataType data;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]{return (this->resources.size() && this->executors.size()) || invalid; });
            if (invalid) return;
            executor = executors.back();
            data     = resources.back();
            executors.pop_back();
            resources.pop_back();
        }
        executor(data);
    }
}

template class ThreadPool<DataPack>;
