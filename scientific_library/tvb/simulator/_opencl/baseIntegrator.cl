__kernel void integrate(__global float *vectors, __global float *coefficients,
                        __global float *ans,
                        int    size,    int lenOfVec){

    int id = get_global_id(0), n = get_global_size(0);

    for(int i = 0; i < size; i++){
        ans[id] += vectors[i*lenOfVec+id];
    }

}