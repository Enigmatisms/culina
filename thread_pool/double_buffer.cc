/**
 * Double buffer (thread pool like)
 * 
 * @author: Qianyue He
 * @date:   2024-3-19
*/

#include <array>
#include <mutex>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>
#include <condition_variable>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_int_distribution<int> time_dis(100, 150);
static std::uniform_int_distribution<int> name_dis(0, 7);
static const std::array<std::string, 8> job_names = {
    "applyGaussianBlur", "detectEdges", "enhanceContrast", 
    "resizeImage", "adjustBrightness", "segmentForeground", 
    "removeNoise", "morphTransform"
};

struct JobSlot {
    JobSlot(): valid(false), delay(0), job_id(-1) {}
    JobSlot(std::string name, int delay, int job_id = 0): job_name(name), valid(true), delay(delay), job_id(job_id) {}

    std::string export_job() {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        valid = false;
        return job_name;
    }

    std::string job_name;
    bool valid;
    int delay;
    int job_id;
};

/**
 * Multi-threading double buffer implementation
*/
class DoubleBuffer {
private:
    // one for read and one for write, use swap
    std::array<JobSlot, 2> slots;
    std::condition_variable_any cv;
    std::timed_mutex mtx;

    std::fstream file;

    std::chrono::system_clock::time_point start;
public:
    DoubleBuffer() {
        start = std::chrono::system_clock::now();
        file.open("./process.log", std::ios::out);
    }

    ~DoubleBuffer() {
        file.close();
    }

    void acquire_job(int acc_factor = 1) {
        static int job_id = 0;
        int acq_delay = time_dis(gen), job_delay = time_dis(gen);       // acquisition is faster
        if (acc_factor > 0) acq_delay >>= acc_factor;
        std::this_thread::sleep_for(std::chrono::milliseconds(acq_delay));
        int name_index = name_dis(gen);
        slots[0] = JobSlot(job_names[name_index], job_delay, ++job_id);

        std::unique_lock lock(mtx, std::chrono::milliseconds(20));
        if (lock.owns_lock()) {
            std::swap(slots[0], slots[1]);
            cv.notify_one();
        }
    }

    // do not discard any incoming job, so the lock timeout is pretty long 
    void acquire_job_no_skip() {
        static int job_id = 0;
        int acq_delay = time_dis(gen) >> 1, job_delay = time_dis(gen);       // acquisition is faster
        std::this_thread::sleep_for(std::chrono::milliseconds(acq_delay));
        int name_index = name_dis(gen);
        slots[0] = JobSlot(job_names[name_index], job_delay, ++job_id);

        std::unique_lock lock(mtx, std::chrono::seconds(1));
        if (lock.owns_lock()) {
            std::swap(slots[0], slots[1]);
            cv.notify_one();
        } else {
            printf("Warning: job acquisition lock expired unexpectedly.\n");
        }
    }

    void process_job() {
        std::unique_lock lock(mtx, std::defer_lock);
        lock.lock();
        cv.wait(lock, [this] {
            return this->slots[1].valid;
        });             // break from blocking, acquire lock once again
        std::this_thread::sleep_for(std::chrono::milliseconds(slots[1].delay));
        file << "Job ("<< slots[1].job_id << "): <" << slots[1].job_name << ">, processing time: " << slots[1].delay << "ms." << std::endl;
        slots[1].valid = false;
    }

    void multi_thread_processing(int acc_factor, int time_out_sec = 5) {
        bool run = true;
        std::thread th([this, acc_factor, &run] {
            while (run) {
                acquire_job(acc_factor);
            }
        });
        th.detach();
        while (std::chrono::system_clock::now() - start < std::chrono::seconds(time_out_sec)) {
            process_job();
        }
        run = false;
    }

    void single_thread_processing(int acc_factor, int time_out_sec = 5) {
        static int job_id = 0;
        while (std::chrono::system_clock::now() - start < std::chrono::seconds(time_out_sec)) {
            int acq_delay = time_dis(gen) >> acc_factor, job_delay = time_dis(gen);       // acquisition is faster
            std::this_thread::sleep_for(std::chrono::milliseconds(acq_delay));
            int name_index = name_dis(gen);
            slots[0] = JobSlot(job_names[name_index], job_delay, ++job_id);
            std::swap(slots[0], slots[1]);

            std::this_thread::sleep_for(std::chrono::milliseconds(slots[1].delay));
            file << "Job ("<< slots[1].job_id << "): <" << slots[1].job_name << ">, processing time: " << slots[1].delay << "ms." << std::endl;
            slots[1].valid = false;
        }
    }
};

int main(int argc, char** argv) {
    DoubleBuffer buffer;
    int acc_factor = 0;
    if (argc > 1)
        acc_factor = std::max(std::min(2, atoi(argv[1])), 0);
    buffer.single_thread_processing(acc_factor);
    return 0;
}