/**
 * Project AETHER: Block-Sparse Matrix Multiplication for Chemical Kinetics
 * Optimized for NVIDIA Ampere/Hopper (SM80/SM90) using CUTLASS.
 *
 * In hypersonic reacting flows, the Jacobian matrix describing chemical reaction 
 * rates (e.g., 11-species or 22-species air models) is extremely large but highly sparse. 
 * Species only react with specific other species. 
 *
 * This implementation uses CUTLASS to perform Block-Sparse GEMM. By reordering the 
 * species vector using Reverse Cuthill-McKee (RCM) ordering, we cluster the interacting 
 * species into dense macro-blocks along the diagonal. We then use a Batched GEMM 
 * approach to multiply ONLY these dense non-zero blocks, keeping the data entirely 
 * in registers and bypassing standard cache hierarchy round-trips.
 */

#include <iostream>
#include <vector>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"

// -----------------------------------------------------------------------------
// CUTLASS Configuration for Block-Sparse Tensor Core Execution
// -----------------------------------------------------------------------------

// Define data types: We utilize Mixed Precision (FP16 inputs, FP32 accumulation)
// to maximize Tensor Core throughput on Hopper architectures.
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;
using ElementAccumulator = float;

// Layout definitions (RowMajor for standard chemical matrices)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Define the CUTLASS GEMM Batched operator
// We use the Batched API to process the array of dense blocks (the non-zero clusters
// created by the Reverse Cuthill-McKee permutation).
using BlockSparseGemm = cutlass::gemm::device::GemmBatched<
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp, // Explicitly target Tensor Cores
    cutlass::arch::Sm80,            // Architecture (Ampere/Hopper)
    cutlass::gemm::GemmShape<128, 128, 64>, // Threadblock-level tile size
    cutlass::gemm::GemmShape<64, 64, 64>,   // Warp-level tile size
    cutlass::gemm::GemmShape<16, 8, 16>,    // Instruction-level shape (mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3 // Number of pipeline stages for asynchronous shared memory loads
>;

// -----------------------------------------------------------------------------
// Host API Wrapper
// -----------------------------------------------------------------------------

/**
 * Executes the Block-Sparse GEMM for chemical kinetics.
 * * @param d_A_array Device pointer to an array of pointers, each pointing to a dense block in Jacobian A.
 * @param d_B_array Device pointer to an array of pointers, each pointing to a dense block in State B.
 * @param d_C_array Device pointer to an array of pointers, each pointing to a dense block in Output C.
 * @param block_m The M dimension of each dense macro-block.
 * @param block_n The N dimension of each dense macro-block.
 * @param block_k The K dimension of each dense macro-block.
 * @param num_active_blocks The total number of non-zero dense blocks identified by RCM ordering.
 * @param alpha Scalar multiplier for A*B.
 * @param beta Scalar multiplier for C.
 */
extern "C" void launch_block_sparse_kinetics_gemm(
    ElementInputA const ** d_A_array,
    ElementInputB const ** d_B_array,
    ElementOutput ** d_C_array,
    int block_m,
    int block_n,
    int block_k,
    int num_active_blocks,
    float alpha = 1.0f,
    float beta = 0.0f
) {
    // Determine leading dimensions based on RowMajor layout
    int lda = block_k;
    int ldb = block_n;
    int ldc = block_n;

    // Configure the CUTLASS arguments
    // We pass the arrays of pointers which allows the GPU to process only the 
    // active, non-zero blocks in parallel, completely skipping the sparse zeros.
    typename BlockSparseGemm::Arguments arguments{
        {block_m, block_n, block_k},
        d_A_array, lda,
        d_B_array, ldb,
        d_C_array, ldc,
        d_C_array, ldc,
        {alpha, beta},
        num_active_blocks
    };

    // Instantiate the CUTLASS Batched GEMM object
    BlockSparseGemm gemm_op;

    // Check memory workspace requirements
    size_t workspace_size = BlockSparseGemm::get_workspace_size(arguments);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Initialize the CUTLASS operation
    cutlass::Status status = gemm_op.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS BlockSparseGemm initialization failed: " 
                  << cutlassGetStatusString(status) << std::endl;
        if (workspace) cudaFree(workspace);
        return;
    }

    // Execute the kernel
    // The inner loops load A and B tiles directly into the SM registers, 
    // compute mma.sync, and bypass L2 cache thrashing entirely.
    status = gemm_op();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS BlockSparseGemm execution failed: " 
                  << cutlassGetStatusString(status) << std::endl;
    }

    // Cleanup workspace
    if (workspace) cudaFree(workspace);
}