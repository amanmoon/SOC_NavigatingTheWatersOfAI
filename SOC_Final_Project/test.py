import torch
import time

# Set the matrix size
matrix_size = 15555

# Generate random matrices
a_cpu = torch.randn(matrix_size, matrix_size)
b_cpu = torch.randn(matrix_size, matrix_size)

# Move matrices to GPU if available
for _ in range(1000000):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)

        # Perform matrix multiplication on GPU and measure time
        start_time_gpu = time.time()
        result_gpu = torch.mm(a_gpu, b_gpu)
        end_time_gpu = time.time()
        time_taken_gpu = end_time_gpu - start_time_gpu
        print(f"Matrix multiplication on GPU took {time_taken_gpu:.6f} seconds.")

    else:
        print("GPU not available. Skipping GPU matrix multiplication.")

    # Perform matrix multiplication on CPU and measure time
    start_time_cpu = time.time()
    result_cpu = torch.mm(a_cpu, b_cpu)
    end_time_cpu = time.time()
    time_taken_cpu = end_time_cpu - start_time_cpu
    print(f"Matrix multiplication on CPU took {time_taken_cpu:.6f} seconds.")
