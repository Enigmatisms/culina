#pragma once

/**
 * A dummy data pack struct
*/
struct DataPack {
    int a;
    int b;
    int c;
    int job_idx;

    DataPack(int a = 0, int b = 0, int c = 0, int job = 0): a(a), b(b), c(c), job_idx(job) {} 
};

