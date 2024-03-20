#pragma once

/**
 * My thread pool implementation
 * @author: Qianyue He
 * @date:   2024.3.13
*/

#include <mutex>
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>
#include "data_pack.hpp"

template <typename DataType>
class ThreadPool {
public:
    ThreadPool(size_t num_workers);

    ~ThreadPool();
public:
    // put some work to be processed (should I pass in the function for callback?)
    // If I need to pass in the callback function pointer, it means that there should be
    // type constraints for the function pointer... which might be difficult to impose
    void enQueue(std::function<void(DataType&)> executor, DataType&& data);

    void executeWork();

    void shut_down();

    void get_queue_status() const {
        printf("Works in the queue: %lu\n", resources.size());
    }
private:
    bool invalid;

    std::mutex mtx;
    std::condition_variable cv;

    std::deque<DataType> resources;
    std::deque<std::function<void(DataType&)>> executors;

    std::vector<std::thread> thread_pool;
};