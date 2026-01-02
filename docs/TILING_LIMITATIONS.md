# TinyTPU: Tiled Matrix Multiplication Limitations

**Current Status**: The TinyTPU design, as of current implementation, is **not** inherently ready for efficient and accurate Tiled Matrix Multiplication.

### Why Tiled Matmul is Not Supported:

Tiled Matrix Multiplication typically involves breaking down large matrices into smaller blocks (tiles), computing the product of these tiles, and then accumulating the partial products in high precision to form the final result. The current TinyTPU architecture presents several limitations that prevent this:

1.  **No High-Precision Partial Sum Accumulation Path**:
    *   **Systolic Array Limitation**: The `pe.sv` and `systolic.sv` modules are designed for a single pass. Specifically, the partial sum input for the top row of Processing Elements (`pe_psum_in` in `src/systolic.sv`) is hardcoded to `32'b0`. This means the systolic array cannot accept or accumulate partial sums from a previous pass, which is fundamental for multi-tile accumulation within the array.
    *   **External Accumulation Requirement**: Without an internal accumulation path, any tiling would necessitate reading the full output matrix from the systolic array, storing it in memory, and then adding subsequent tile products to it via software or an external accumulator.

2.  **Premature Truncation in Vector Processing Unit (VPU)**:
    *   The `vpu.sv` module immediately processes the 32-bit output of the systolic array. In the `gen_requant` block, the 32-bit `vpu_data_in` is either right-shifted by 8 bits (for Q8.8 mode) or directly truncated to its lower 16 bits (`vpu_data_in[i][15:0]`).
    *   This "Premature Truncation" means that the full 32-bit precision of the accumulated products from the systolic array is lost *before* any opportunity for further high-precision accumulation from other tiles. If tiling were to be performed externally (e.g., in software by repeatedly loading and accumulating 16-bit results), it would lead to significant accuracy degradation due to accumulating quantized/truncated values rather than full-precision intermediate sums.

3.  **Unified Buffer and VPU Accumulator Limitations**:
    *   **Unified Buffer (UB)**: The Unified Buffer (UB) is constrained to a 16-bit data width. This means even if the VPU could temporarily hold 32-bit values, storing and retrieving them from the UB for accumulation would involve splitting and rejoining 16-bit words, adding complexity and potential overhead.
    *   **VPU Modules**: The VPU's child modules (e.g., `bias_child`) are designed for specific element-wise operations (like adding a bias) on 16-bit data. There is no dedicated 32-bit (or higher) accumulator module within the VPU that could read a previous partial sum from memory and add it to the current 32-bit systolic array output before requantization.

### Changes Required for Tiled Matmul Support:

To enable proper and accurate Tiled Matrix Multiplication, the following architectural modifications would be necessary:

1.  **Systolic Array Input for Partial Sums**:
    *   Modify `src/systolic.sv` and `src/pe.sv` to accept an optional 32-bit `pe_psum_in` for the top-most row of PEs, allowing for accumulation of external partial sums.
    *   The control unit would need to manage when to feed an initial zero for the first tile, and when to feed previous partial sums for subsequent tiles.

2.  **Delayed and Controlled Requantization**:
    *   The VPU (`src/vpu.sv`) would need to be redesigned to allow the 32-bit output from the systolic array to either:
        *   Be accumulated with a previous 32-bit partial sum (read from memory) *before* any requantization/truncation.
        *   Be stored directly in 32-bit form into a dedicated buffer, if the UB is not upgraded to 32-bit.
    *   The `requant_out` step should ideally be moved to occur only after all partial products for a given output matrix element have been accumulated.

3.  **High-Precision Accumulator in VPU (or dedicated module)**:
    *   A dedicated module would be needed to handle the 32-bit accumulation of partial results, potentially reading from and writing back to a temporary 32-bit memory (if available).

Without these changes, any attempt at tiling large matrices would either result in incorrect computations (due to the hardcoded zero-initialization of partial sums) or significant loss of numerical precision (due to premature truncation/quantization).