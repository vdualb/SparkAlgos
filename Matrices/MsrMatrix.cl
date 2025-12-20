#define real double
#define real4 float4

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void MsrMul(
    global const real *mat,
    global const real *di,
    global const int *aptr,
    global const int *jptr,
    
    const int n,
    
    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    if (row < n)
    {
        int start = aptr[row];
        int stop = aptr[row + 1];
        real dot = di[row]*v[row];
        for (int a = start; a < stop; a++)
        {
            dot += mat[a]*v[jptr[a]];
        }
        res[row] = dot;
    }
}

kernel void LMul(
    global const real *mat,
    global const real *di,
    global const int *aptr,
    global const int *jptr,
    
    const int n,

    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    if (row < n)
    {
        int start = aptr[row];
        int stop = aptr[row + 1];
        real dot = di[row]*v[row];
        for (int a = start; a < stop && jptr[a] < row; a++)
        {
            dot += mat[a]*v[jptr[a]];
        }
        res[row] = dot;
    }
}

kernel void UMul(
    global const real *mat,
    global const real *di,
    global const int *aptr,
    global const int *jptr,
    
    const int n,

    global const real *v,
    global real *res)
{
    size_t row = get_global_id(0);

    if (row < n)
    {
        int start = aptr[row];
        int stop = aptr[row + 1];
        real dot = di[row]*v[row];
        for (int a = start; a < stop; a++)
        {
            if (jptr[a] > row) {
                dot += mat[a]*v[jptr[a]];
            }
        }
        res[row] = dot;
    }
}

// https://github.com/viennacl/viennacl-dev/blob/dc552a8d4bba2bb028a346dc611deea0c67e3eed/viennacl/linalg/cuda/sparse_matrix_operations_solve.hpp#L257
// modified for MSR format
kernel void InvLMul(
    global const real *mat,
    global const real *di,
    global const int *aptr,
    global const int *jptr,
    
    const int n,

    global real *res)
{
    local unsigned int jptr_buffer[128];
    local real element_buffer[128];
    local real vector_buffer[128];
  
    unsigned int nnz = aptr[n];
    unsigned int current_row = 0;
    unsigned int row_at_window_start = 0;
    real current_vector_entry = res[0];
    unsigned int loop_end = (nnz / get_local_size(0) + 1) * get_local_size(0);
  
    unsigned k = 0;
    unsigned a0 = aptr[0];
    unsigned a1 = aptr[1];
    unsigned ia = a0;
    
    for (unsigned int i = get_local_id(0); i < loop_end; i += get_local_size(0))
    {
        //load into shared memory (coalesced access):
        if (i < nnz)
        {
            element_buffer[get_local_id(0)] = mat[i];
            unsigned int tmp = jptr[i];
            jptr_buffer[get_local_id(0)] = tmp;
            vector_buffer[get_local_id(0)] = res[tmp];
        }
  
        barrier(CLK_LOCAL_MEM_FENCE);
    
        //now a single thread does the remaining work in shared memory:
        if (get_local_id(0) == 0)
        {
            // printf("2. i = %d", i);
            k = 0;
            
            while (1)
            {
                // iterate over rows, while k is inside the window
                for (; ia < a1 && k < get_local_size(0); ia++, k++) {
                    
                    if (jptr_buffer[k] < current_row) {
                        if (jptr_buffer[k] < row_at_window_start) //use buffered data
                        {
                            // printf("3. %f -= %f * %f, k = %d\n", current_vector_entry, element_buffer[k], vector_buffer[k], k);
                            current_vector_entry -= element_buffer[k] * vector_buffer[k];
                        }
                        else if (jptr_buffer[k] < current_row) //use recently computed results
                        {
                            // printf("4. %f -= %f * %f, k = %d\n", current_vector_entry, element_buffer[k], res[jptr_buffer[k]], k);
                            current_vector_entry -= element_buffer[k] * res[jptr_buffer[k]];
                        }
                    }
                }
                
                if (ia == a1) {
                    // printf("5. %f = %f / %f, curr_row = %d\n", res[current_row], current_vector_entry, di[current_row], current_row);
                    res[current_row] = current_vector_entry / di[current_row];
                    if (current_row < n - 1) {
                        current_row++;
                        
                        current_vector_entry = res[current_row];
                        a0 = aptr[current_row];
                        a1 = aptr[current_row + 1];
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } // while (1)

            row_at_window_start = current_row;
        } // if (get_local_id(0) == 0)
    
        barrier(CLK_GLOBAL_MEM_FENCE);
    } //for i
}

// https://github.com/viennacl/viennacl-dev/blob/dc552a8d4bba2bb028a346dc611deea0c67e3eed/viennacl/linalg/cuda/sparse_matrix_operations_solve.hpp#L257
kernel void InvUMul(
    global const real *mat,
    global const real *di,
    global const int *aptr,
    global const int *jptr,
    
    const int n,

    global real *res)
{
    local unsigned int jptr_buffer[128];
    local real element_buffer[128];
    local real vector_buffer[128];
    
    unsigned int nnz = aptr[n];
    unsigned int current_row = n-1;
    unsigned int row_at_window_start = n-1;
    real current_vector_entry = res[n-1];
    unsigned int loop_end = ( (nnz - 1) / get_local_size(0)) * get_local_size(0);
    unsigned int next_row = aptr[n-1];

    int a0 = aptr[n]-1;
    int a1 = aptr[n-1];
    int k = (nnz-1) % get_local_size(0);
    int ia = a0;
    
    // if (get_local_id(0) == 0) printf("nnz = %d", nnz);
    
    unsigned int i = loop_end + get_local_id(0);
    while (1)
    {
        //load into shared memory (coalesced access):
        if (i < nnz)
        {
            element_buffer[get_local_id(0)] = mat[i];
            unsigned int tmp = jptr[i];
            jptr_buffer[get_local_id(0)] = tmp;
            vector_buffer[get_local_id(0)] = res[tmp];
        }
    
        barrier(CLK_GLOBAL_MEM_FENCE);
    
        //now a single thread does the remaining work in shared memory:
        if (get_local_id(0) == 0)
        {
            if (i != loop_end)
            {
                k = get_local_size(0) - 1;
            }
            
            while (1)
            {
                // iterate over rows, while k is inside the window
                while (ia >= a1 && k >= 0) {
                    if (jptr_buffer[k] > current_row) {
                        if (jptr_buffer[k] > row_at_window_start) //use buffered data
                        {
                            // printf("3. %f -= %f * %f, k = %d\n", current_vector_entry, element_buffer[k], vector_buffer[k], k);
                            current_vector_entry -= element_buffer[k] * vector_buffer[k];
                        }
                        else if (jptr_buffer[k] > current_row) //use recently computed results
                        {
                            // printf("4. %f -= %f * %f, k = %d\n", current_vector_entry, element_buffer[k], res[jptr_buffer[k]], k);
                            current_vector_entry -= element_buffer[k] * res[jptr_buffer[k]];
                        } else {
                            // printf("Unexpected");
                        }
                    } else {
                        // printf("Skip, ia = %d, k = %d", ia, k);
                    }
                    ia--;
                    k--;
                }
                
                if (ia < a1) {
                    // printf("5. %f = %f / %f, curr_row = %d\n", res[current_row], current_vector_entry, di[current_row], current_row);
                    res[current_row] = current_vector_entry / di[current_row];
                    if (current_row >= 1) {
                        current_row--;
                        
                        current_vector_entry = res[current_row];
                        a0 = aptr[current_row + 1]-1;
                        a1 = aptr[current_row];
                        ia = a0;
                    } else {
                        break;
                    }
                } else {
                    // printf("Page end, ia = %d, k = %d", ia, k);
                    break;
                }
            } // while (1)
            
            row_at_window_start = current_row;
        } // if (get_local_id(0) == 0)
    
        barrier(CLK_GLOBAL_MEM_FENCE);
    
        if (i < get_local_size(0))
            break;
    
        i -= get_local_size(0);
    } //for i
}
