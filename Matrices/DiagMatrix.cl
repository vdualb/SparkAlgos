#define real double
#define real4 float4

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void DiagMul(
    global const real *ld3,
    global const real *ld2,
    global const real *ld1,
    global const real *ld0,
    
    global const real *di,
    
    global const real *rd0,
    global const real *rd1,
    global const real *rd2,
    global const real *rd3,

    const int n,
    const int gap,
    
    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    if (row < n)
    {
        real dot = 0;
        
        int t = row - 3 - gap;
        if (t >= 0) dot += ld3[t] * v[t];
        t = row - 2 - gap;
        if (t >= 0) dot += ld2[t] * v[t];
        t = row - 1 - gap;
        if (t >= 0) dot += ld1[t] * v[t];
        t = row - 1;
        if (t >= 0) dot += ld0[t] * v[t];
        
        dot += di[row] * v[row];

        t = row+1;
        if (t < n) dot += rd0[row] * v[t];
        t = row+1+gap;
        if (t < n) dot += rd1[row] * v[t];
        t = row+2+gap;
        if (t < n) dot += rd2[row] * v[t];
        t = row+3+gap;
        if (t < n) dot += rd3[row] * v[t];
        
        res[row] = dot;
    }
}

kernel void LMul(
    global const real *ld3,
    global const real *ld2,
    global const real *ld1,
    global const real *ld0,
    
    global const real *di,

    const int n,
    const int gap,
    
    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    if (row < n)
    {
        real dot = di[row] * v[row];
        
        int t = row - 3 - gap;
        if (t >= 0) dot += ld3[t] * v[t];
        t = row - 2 - gap;
        if (t >= 0) dot += ld2[t] * v[t];
        t = row - 1 - gap;
        if (t >= 0) dot += ld1[t] * v[t];
        t = row - 1;
        if (t >= 0) dot += ld0[t] * v[t];
        
        res[row] = dot;
    }
}

kernel void UMul(
    global const real *rd3,
    global const real *rd2,
    global const real *rd1,
    global const real *rd0,
    
    global const real *di,

    const int n,
    const int gap,
    
    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    if (row < n)
    {
        real dot = di[row] * v[row];
        
        int t = row+1;
        if (t < n) dot += rd0[row] * v[t];
        t = row+1+gap;
        if (t < n) dot += rd1[row] * v[t];
        t = row+2+gap;
        if (t < n) dot += rd2[row] * v[t];
        t = row+3+gap;
        if (t < n) dot += rd3[row] * v[t];
        
        res[row] = dot;
    }
}

// local work size: 4
// global work size: 4
kernel void InvLMul(
    global const real *ld3,
    global const real *ld2,
    global const real *ld1,
    global const real *ld0,
    
    global const real *di,

    const int n,
    const int gap,
    
    global real *res)
{
    size_t locid = get_local_id(0);
    global const real *locdiags[] = {ld0, ld1, ld2, ld3};
    // printf("1. This should get called 4 times");

    for (int i = 0; i < n; i++)
    {
        if (locid == 0)
        {
            // printf("2. %lf /= %lf, i = %d\n", res[i], di[i], i);
            res[i] /= di[i];
            real new_res = res[i];
            // int t = i + 1;
            // if (t < n) {
            //     // printf("3. %lf -= %lf * %lf, i = %d, t = %d\n", res[t], locdiags[locid][i], res[i], i, t);
            //     res[t] -= locdiags[locid][i] * new_res;
            // } 
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE); 
        
        real new_res = res[i];

        if (locid == 0) {
            int t = i + 1;
            if (t < n) {
                // printf("3. %lf -= %lf * %lf, i = %d, t = %d\n", res[t], locdiags[locid][i], res[i], i, t);
                res[t] -= locdiags[locid][i] * new_res;
            }    
        }
        
        // thread 1 2 3
        if (locid == 1 || locid == 2 || locid == 3)
        {
            int t = i + locid + gap;
            if (t < n) {
                // printf("4. %lf -= %lf * %lf, i = %d, t = %d\n", res[t], locdiags[locid][i], res[i], i, t);
                res[t] -= locdiags[locid][i] * new_res;
            }
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// local work size: 4
// global work size: 4
kernel void InvUMul(
    global const real *rd3,
    global const real *rd2,
    global const real *rd1,
    global const real *rd0,
    
    global const real *di,

    const int n,
    const int gap,
    
    global real *res)
{
    size_t locid = get_local_id(0);
    global const real *locdiags[] = {rd0, rd1, rd2, rd3};
    // printf("1. This should get called 4 times");
    
    // TODO: work on a local res copy
    for (int i = n-1; i >= 0; i--)
    {
        if (locid == 0)
        {
            // printf("2. %lf /= %lf, i = %d\n", res[i], di[i], i);
            res[i] /= di[i];
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE); // TODO: this will change to CLK_LOCAL_MEM_FENCE
        
        if (locid == 0) {
            int t = i - 1;
            if (t >= 0) {
                // printf("3. %lf -= %lf * %lf, i = %d, t = %d\n", res[t], locdiags[locid][t], res[i], i, t);
                res[t] -= locdiags[locid][t] * res[i];
            }
        }
        
        // thread 1 2 3
        if (locid == 1 || locid == 2 || locid == 3)
        {
            int t = i - locid - gap;
            if (t >= 0) {
                // printf("4. %lf -= %lf * %lf, i = %d, t = %d\n", res[t], locdiags[locid][t], res[i], i, t);
                res[t] -= locdiags[locid][t] * res[i];
            }
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
