#include <chrono>
#include <random>
#include "./thread_pool.hpp"


int rand_int() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(1, 100);
    return dis(gen);
}

void sum_mod_data_pack(DataPack& data_pack) {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    printf("Job(%d) -- Work (%d + %d) %% %d = %d\n", data_pack.job_idx,
        data_pack.a, data_pack.b, data_pack.c, (data_pack.a + data_pack.b) % (data_pack.c));
}

void prod_sum_data_pack(DataPack& data_pack) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    printf("Job(%d) -- Work (%d + %d) * %d = %d\n", data_pack.job_idx,
        data_pack.a, data_pack.b, data_pack.c, (data_pack.a + data_pack.b) * (data_pack.c));
}


int main(int argc, char** argv) {
    size_t num_threads = 108;
    if (argc > 1)
        num_threads = std::max(atoi(argv[1]), 1);
    ThreadPool<DataPack> pool(num_threads);
    for (int i = 0; i < 100000; i++) {
        auto func = i & 1 ? prod_sum_data_pack : sum_mod_data_pack;
        pool.enQueue(func, {rand_int(), rand_int(), rand_int(), i});
        if (i == 10000)
            pool.shut_down();
    }
    return 0;
}