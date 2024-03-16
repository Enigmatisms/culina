#include <chrono>

class TicToc {
private:
    std::chrono::system_clock::time_point tp;
public:
    void tic() {
        tp = std::chrono::system_clock::now();
    }

    double toc() const {
        auto dur = std::chrono::system_clock::now() - tp;
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        return static_cast<double>(count) / 1e3;
    }
};