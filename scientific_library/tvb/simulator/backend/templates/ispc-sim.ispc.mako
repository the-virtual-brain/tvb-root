task void dfun_eval_chunk() {
    for (uniform int i=0; i<chunksize; i++) {
        varying lc1 = csr_row(...);
        varying sc1 = csr_row(...);
        varying dx = derivatives(...);
    }
}

export void dfun_eval() { }

// Heun step functions proceed with certain parts
task void heun_stage1_chunk() {}
export void heun_stage1() {}

task void heun_stage2_chunk() {}
export void heun_stage2() {}

task void heun_finalize_chunk() {}
export void heun_finalize() {}

// but all the task launchers have same structure -> template it

#define task_launcher(name, task, ...) \
export void name(__VA_ARGS__) { \
    for (uniform int i=0; i<100; i++) { \
        launch task(i, __VA_ARGS__); \
    } \
    sync; \
}

task_launcher(foo, bar, int x, int y, int z)

// cpp is not up to the task here
