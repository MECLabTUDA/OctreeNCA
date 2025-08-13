#include <torch/extension.h>
#include <curand_kernel.h>

#define C 16
#define HIDDEN 64
// must remain 3 for now
#define KERNEL_SIZE 3
#define FIRE_RATE 1.0f

typedef uint32_t index_type;

__device__ inline index_type unravel_index(index_type channel, index_type x, index_type y, index_type z, index_type W, index_type H, index_type D) {
    return channel * W * H * D + y * W * D + x * D + z;
}

__device__ inline index_type unravel_index_kernel(index_type i_x, index_type i_y, index_type i_z, index_type channel) {
    return channel * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE + i_y * KERNEL_SIZE * KERNEL_SIZE + i_x * KERNEL_SIZE + i_z;
}

__device__ inline index_type unravel_index_linear0(index_type in, index_type out) {
    return 2*C * out + in;
}

__device__ inline index_type unravel_index_linear1(index_type in, index_type out) {
    return HIDDEN * out + in;
}

__global__ void nca3d_cuda_kernel(
    float * new_state,
    const float* state, 
    const float* conv_weight, 
    const float* conv_bias, 
    const float* fc0_weight, 
    const float* fc0_bias, 
    const float* fc1_weight,
    int W, 
    int H,
    int D,
    int out_C) {
    
    
    index_type tx = threadIdx.x + blockIdx.x * blockDim.x;
    index_type ty = threadIdx.y + blockIdx.y * blockDim.y;
    index_type tz = threadIdx.z + blockIdx.z * blockDim.z;

    float out_conv[C];
    float hidden[HIDDEN];


    if (tx < W && ty < H && tz < D) {

        if(new_state[unravel_index(0, tx, ty, tz, W, H, D)] < 0.5) {
            for(int out = 0; out < C; out++) {
                new_state[unravel_index(out, tx, ty, tz, W, H, D)] = state[unravel_index(out, tx, ty, tz, W, H, D)];
            }
            return;
        }

        for(int current_channel = 0; current_channel < C; current_channel++) {
            out_conv[current_channel] = 0.0f;
            for(int i_y = 0; i_y < KERNEL_SIZE; i_y++) {
                for(int i_x = 0; i_x < KERNEL_SIZE; i_x++) {
                    for(int i_z = 0; i_z < KERNEL_SIZE; i_z++){

                        // reflect padding
                        index_type input_x = tx + i_x-1;
                        index_type input_y = ty + i_y-1;
                        index_type input_z = tz + i_z-1;
                        if(input_x < 0) {
                            input_x = 1;
                        }
                        else if(input_x >= W) {
                            input_x = W - 2;
                        }
                        if(input_y < 0) {
                            input_y = 1;
                        }
                        else if(input_y >= H) {
                            input_y = H - 2;
                        }
                        if(input_z < 0) {
                            input_z = 1;
                        }
                        else if(input_z >= D) {
                            input_z = D - 2;
                        }



                        index_type input_index = unravel_index(current_channel, input_x, input_y, input_z, W, H, D);
                        index_type kernel_index = unravel_index_kernel(i_x, i_y, i_z, current_channel);
                        out_conv[current_channel] += state[input_index] * conv_weight[kernel_index];
                    }
                }
            }
            out_conv[current_channel] += conv_bias[current_channel];
        }

        for (int out = 0; out < HIDDEN; out++) {
            hidden[out] = 0.0f;
            for(int in = 0; in < C; in++) {
                index_type input_index = unravel_index(in, tx, ty, tz, W, H, D);
                hidden[out] += state[input_index] * fc0_weight[unravel_index_linear0(in, out)];
            }
            for(int in = 0; in < C; in++) {
                hidden[out] += out_conv[in] * fc0_weight[unravel_index_linear0(in+C, out)];
            }
            hidden[out] += fc0_bias[out];
            
            //apply ReLU
            hidden[out] = hidden[out] > 0 ? hidden[out] : 0;
        }
        //for(int i=0;i<HIDDEN;i++) {
        //    temp[unravel_index(i, tx, ty, W, H)] = hidden[i];
        //}
        //return;

        for (int out = 0; out < C; out++) {
            index_type out_index = unravel_index(out, tx, ty, tz, W, H, D);
            float res = 0.0f;
            if(out < C - out_C){
                new_state[out_index] = state[out_index];
            }
            else{
                for(int in = 0; in < HIDDEN; in++) {
                    res += hidden[in] * fc1_weight[unravel_index_linear1(in, out - (C - out_C))];
                }
                new_state[out_index] = res + state[unravel_index(out, tx, ty, tz, W, H, D)];
            }
        }
        //new_state[unravel_index(0, tx, ty, W, H)] = hidden[0];
        //new_state[unravel_index(1, tx, ty, W, H)] = hidden[1];
        //return;
        
    }
}


torch::Tensor nca3d_cuda(
    torch::Tensor& new_state,
    const torch::Tensor& state, 
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& fc0_weight,
    const torch::Tensor& fc0_bias,
    const torch::Tensor& fc1_weight
    ) {
    TORCH_CHECK(new_state.is_cuda(), "random must be a CUDA tensor");
    TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    TORCH_CHECK(fc0_weight.is_cuda(), "fc0_weight must be a CUDA tensor");
    TORCH_CHECK(fc0_bias.is_cuda(), "fc0_bias must be a CUDA tensor");
    TORCH_CHECK(fc1_weight.is_cuda(), "fc1_weight must be a CUDA tensor");


    TORCH_CHECK(new_state.dim() == 5, "random must be B1HWD");
    TORCH_CHECK(state.dim() == 5, "state must be BCHWD");
    TORCH_CHECK(conv_weight.dim() == 5, "conv must be 2D");
    TORCH_CHECK(conv_bias.dim() == 1,"conv must be 1D");
    TORCH_CHECK(fc0_weight.dim() == 5, "fc0 must be 2D");
    TORCH_CHECK(fc0_bias.dim() == 1, "fc0 must be 1D");
    TORCH_CHECK(fc1_weight.dim() == 5, "fc1 must be 2D");


    
    TORCH_CHECK(new_state.is_contiguous(), "random must be contiguous");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
    TORCH_CHECK(conv_weight.is_contiguous(), "conv_weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(fc0_weight.is_contiguous(), "fc0_weight must be contiguous");
    TORCH_CHECK(fc0_bias.is_contiguous(), "fc0_bias must be contiguous");
    TORCH_CHECK(fc1_weight.is_contiguous(), "fc1_weight must be contiguous");

    int B = state.size(0);
    //int C = state.size(1);
    int H = state.size(2);
    int W = state.size(3);
    int D = state.size(4);

    int out_C = fc1_weight.size(0);

    TORCH_CHECK(new_state.numel() == state.numel(), "new_state too large");
    TORCH_CHECK(state.numel() < std::numeric_limits<index_type>::max(), "state too large");


    TORCH_CHECK(new_state.size(0) == B, "Random batch size mismatch");
    TORCH_CHECK(new_state.size(1) == C, "Random size mismatch");
    TORCH_CHECK(new_state.size(2) == H, "Random height size mismatch");
    TORCH_CHECK(new_state.size(3) == W, "Random width size mismatch");
    TORCH_CHECK(new_state.size(4) == D, "Random depth size mismatch");

    TORCH_CHECK(state.size(1) == C, "State channel size mismatch");

    //int nca_hidden_size = fc0_weight.size(0);
    TORCH_CHECK(fc0_weight.size(0) == HIDDEN, "FC0 weight size mismatch");

    TORCH_CHECK(B == 1, "Only batch size 1 is supported");
    TORCH_CHECK(conv_weight.size(0) == C, "Conv weight size mismatch");
    TORCH_CHECK(conv_weight.size(1) == 1, "Conv weight size mismatch");
    TORCH_CHECK(conv_weight.size(2) == KERNEL_SIZE, "Conv weight size mismatch");
    TORCH_CHECK(conv_weight.size(3) == KERNEL_SIZE, "Conv weight size mismatch");

    TORCH_CHECK(conv_bias.size(0) == C, "Conv bias size mismatch");

    TORCH_CHECK(fc0_weight.size(1) == 2*C, "FC0 weight size mismatch");
    TORCH_CHECK(fc0_weight.size(2) == 1, "FC0 weight size mismatch");
    TORCH_CHECK(fc0_weight.size(3) == 1, "FC0 weight size mismatch");
    TORCH_CHECK(fc0_weight.size(4) == 1, "FC0 weight size mismatch");

    TORCH_CHECK(fc0_bias.size(0) == HIDDEN, "FC0 bias size mismatch");

    TORCH_CHECK(fc1_weight.size(1) == HIDDEN, "FC1 weight size mismatch");
    TORCH_CHECK(fc1_weight.size(2) == 1, "FC1 weight size mismatch");
    TORCH_CHECK(fc1_weight.size(3) == 1, "FC1 weight size mismatch");
    TORCH_CHECK(fc1_weight.size(4) == 1, "FC1 weight size mismatch");

    //auto new_state = torch::zeros({1, out_C, H, W, D}, state.options());

    dim3 threads(8, 8, 8);
    dim3 blocks((W + threads.x - 1) / threads.x,
                (H + threads.y - 1) / threads.y,
                (D + threads.z - 1) / threads.z);

    nca3d_cuda_kernel<<<blocks, threads>>>(
        new_state.data_ptr<float>(),
        state.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        fc0_weight.data_ptr<float>(),
        fc0_bias.data_ptr<float>(),
        fc1_weight.data_ptr<float>(),
        W, 
        H,
        D,
        out_C);

    return new_state;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nca3d_cuda", &nca3d_cuda, "3D NCA CUDA");
}
