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
    unsigned int next_row = aptr[1];
  
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
            // traverse through all the loaded data:
            for (unsigned int k=0; k<get_local_size(0); ++k)
            {
                if (current_row < n && i+k == next_row) //current row is finished. Write back result
                {
                    res[current_row] = current_vector_entry / di[current_row];
                    ++current_row;
                    if (current_row < n) //load next row's data
                    {
                        next_row = aptr[current_row+1];
                        current_vector_entry = res[current_row];
                    }
                }
        
                if (current_row < n && jptr_buffer[k] < current_row) //substitute
                {
                    if (jptr_buffer[k] < row_at_window_start) //use recently computed results
                    {
                        current_vector_entry -= element_buffer[k] * vector_buffer[k];
                    }
                    else if (jptr_buffer[k] < current_row) //use buffered data
                    {
                        current_vector_entry -= element_buffer[k] * res[jptr_buffer[k]];
                    }
                }
            } // for k
    
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

    if (get_local_id(0) == 0) printf("nnz = %d", nnz);
    
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
            unsigned a0 = aptr[current_row];
            unsigned a1 = aptr[current_row + 1];
            for (unsigned int i2 = a0; i2 < a1; i2++) {
                
            }
            
            // traverse through all the loaded data from back to front:
            for (unsigned int k2=0; k2<get_local_size(0); ++k2)
            {
                unsigned int k = (get_local_size(0) - k2) - 1;
        
                if (i+k >= nnz)
                {
                    continue;
                }
                
                printf("i = %d", i);
        
                if (jptr_buffer[k] > row_at_window_start) //use recently computed results
                {
                    printf("3. %f -= %f * %f, k = %d\n", current_vector_entry, element_buffer[k], vector_buffer[k], k);
                    current_vector_entry -= element_buffer[k] * vector_buffer[k];
                }
                else if (jptr_buffer[k] > current_row) //use buffered data
                {
                    printf("4. %f -= %f * %f, k = %d\n", current_vector_entry, element_buffer[k], res[jptr_buffer[k]], k);
                    current_vector_entry -= element_buffer[k] * res[jptr_buffer[k]];
                }
        
                if (i+k < next_row) //current row is finished. Write back result
                {
                    printf("5. %f = %f / %f, curr_row = %d\n", res[current_row], current_vector_entry, di[current_row], current_row);
                    res[current_row] = current_vector_entry / di[current_row];
                    if (current_row > 0) //load next row's data
                    {
                        --current_row;
                        next_row = aptr[current_row];
                        current_vector_entry = res[current_row];
                    }
                }
            } // for k
        
            row_at_window_start = current_row;
        } // if (get_local_id(0) == 0)
    
        barrier(CLK_GLOBAL_MEM_FENCE);
    
        if (i < get_local_size(0))
            break;
    
        i -= get_local_size(0);
    } //for i
}
