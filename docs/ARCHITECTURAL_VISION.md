# TinyTPU Architectural Evolution: Supporting Multi-Precision Integer Arithmetic

This document outlines the planned architectural changes to the TinyTPU, moving from a pure Q8.8 fixed-point design to a flexible, multi-precision integer-based architecture. The primary goal is to enhance performance and efficiency by aligning with modern AI accelerator design principles, particularly concerning low-bit quantization.

---

## 1. Core Architectural Shift: Fixed-Point to Quantized Integer

The existing TinyTPU design extensively uses a **16-bit Q8.8 fixed-point** format throughout its computational pipeline. While functional, this approach leads to precision loss at every multiplication stage due to immediate truncation/rounding back to 16 bits.

The new approach will pivot to a **Quantized Integer Pipeline**:
*   **Operations:** All multiplications (`INT8 x INT8`, `INT4 x INT4`, etc.) will produce wider, signed pure integer results.
*   **Accumulation:** Processing Elements (PEs) will accumulate these products into **32-bit signed integer accumulators**. Crucially, the **vertical partial sum paths (`pe_psum_in/out` signals in `systolic.sv` and `pe.sv`) will also be widened to 32-bit** to prevent intermediate overflow and preserve precision across the systolic array.
*   **Requantization:** The crucial step of scaling and shifting the accumulated 32-bit values back to a lower precision (e.g., INT8 or INT16) will occur **once, at the end of the MAC operation**, typically within the Vector Processing Unit (VPU). This deferred requantization significantly improves accuracy. The VPU will require additional configuration inputs such as `requant_scale`, `requant_shift`, and `requant_zero_point` to perform this operation correctly.

---

## 2. Introducing Multi-Modal Precision Control (`sys_mode`)

A global `sys_mode` control signal (e.g., a 2-bit or 3-bit input) will be introduced to dynamically configure the TPU's operational precision. This signal will be propagated to the Systolic Array PEs and the VPU.

**Proposed `sys_mode` Definitions:**

*   **`00` - Q8.8 Fixed-Point Mode:**
    *   **Behavior:** Legacy mode. Inputs are interpreted as Q8.8 signed fixed-point numbers.
    *   **PE Operation:** `(A_Q8.8 * W_Q8.8)`. The multiplication result is effectively Q16.16. This full-precision product (32-bit) is directly accumulated into the 32-bit PE accumulator. The final fixed-point shift (e.g., `>>> 8`) will occur during the requantization stage in the VPU, allowing for higher precision accumulation.
    *   **Output:** 32-bit accumulated value (effectively Q16.16).
    *   **Purpose:** Maintain compatibility and provide a higher-precision accumulation option.

*   **`01` - INT16 Mode:**
    *   **Behavior:** Pure 16-bit signed integer multiplication and accumulation.
    *   **PE Operation:** `A_INT16 * W_INT16` (resulting in a 32-bit signed integer product, accumulated in the 32-bit PE accumulator).
    *   **Output:** 32-bit accumulated value. Requantization in VPU will scale/shift back to 16-bit signed INT.
    *   **Purpose:** Baseline integer precision.

*   **`10` - INT8 Packed Mode (W8A8):**
    *   **Concept:** **SIMD Within A Register (SWAR)** for 8-bit signed integers.
    *   **Packing:** Two 8-bit signed input values (A) are packed into a single 16-bit word (`[A_high | A_low]`). Similarly, two 8-bit signed weight values (W) are packed (`[W_high | W_low]`).
    *   **PE Operation:** Performs a "Dot Product" of the packed values: `(A_high_INT8 * W_high_INT8) + (A_low_INT8 * W_low_INT8)`.
    *   **Accumulation:** The sum of two `INT8 x INT8` products (each producing a `INT16` result) is accumulated into the 32-bit PE accumulator.
    *   **Output:** 32-bit accumulated value. Requantization in VPU scales/shifts to 8-bit signed INT.
    *   **Benefit:** **2x throughput** (2 MACs per cycle) for 8-bit operations.

*   **`11` - INT4 Packed Mode (W4A4):**
    *   **Concept:** SWAR for 4-bit signed integers.
    *   **Packing:** Four 4-bit signed input values (A) are packed into a single 16-bit word (`[A3 | A2 | A1 | A0]`). Similarly, four 4-bit signed weight values (W) are packed.
    *   **PE Operation:** Performs a dot product of four pairs: `(A0_INT4 * W0_INT4) + (A1_INT4 * W1_INT4) + (A2_INT4 * W2_INT4) + (A3_INT4 * W3_INT4)`. For this, 4-bit values will be sign-extended to 8-bit *before* multiplication to reuse existing 8x8 multipliers.
    *   **Accumulation:** The sum of four `INT4 x INT4` products (each producing an `INT8` result) is accumulated into the 32-bit PE accumulator.
    *   **Output:** 32-bit accumulated value. Requantization in VPU scales/shifts to 4-bit signed INT.
    *   **Benefit:** **4x throughput** (4 MACs per cycle) for 4-bit operations.

---

## 3. Handling Mixed Precision (W4A8)

While not a direct `sys_mode` setting for a single cycle, a mixed-precision scenario like **W4A8 (4-bit Weights, 8-bit Activations)** can be supported by leveraging the INT8 Packed mode hardware with temporal multiplexing and sign extension:

*   **Bandwidth Mismatch:** The 16-bit Activation bus can carry two 8-bit Activations, while the 16-bit Weight bus can carry four 4-bit Weights. This limits computation to 2 MACs per cycle based on Activations.
*   **"Gearbox" Strategy within PE:**
    1.  **Cycle 1 (Lower Weights):** The PE processes the lower two 4-bit weights from the weight input (sign-extended to 8-bit) and multiplies them with the two 8-bit activations. Accumulates `(A0_INT8 * W0_INT4_ext) + (A1_INT8 * W1_INT4_ext)`.
    2.  **Cycle 2 (Upper Weights):** The PE *retains* the same 16-bit weight input from the previous cycle. It then processes the upper two 4-bit weights (sign-extended to 8-bit) and multiplies them with a *new* pair of two 8-bit activations from the input. Accumulates `(A2_INT8 * W2_INT4_ext) + (A3_INT8 * W3_INT4_ext)`.
*   **Benefit:** Reduces weight memory bandwidth requirements by half, as the weight memory is accessed once every two cycles for this mode. The throughput remains 2 MACs per cycle (limited by A8 input), but memory efficiency is greatly greatly improved.
*   **Sign Extension:** Crucially, all 4-bit weights are **sign-extended to 8-bit** before multiplication to reuse the existing `INT8 x INT8` multiplier hardware.

---

## 4. Implementation Steps (Roadmap)

To achieve this flexible multi-precision architecture, the following refactoring steps are necessary:

1.  **`tpu.sv`:**
    *   Add `input logic [1:0] sys_mode` (or wider) to the top-level module interface.
    *   Pass `sys_mode` to the `systolic` and `vpu` instantiations.

2.  **`systolic.sv`:**
    *   Accept `sys_mode` as an input.
    *   Widen the vertical partial sum data paths (`pe_psum_out` and internal wires between PEs) from `[15:0]` to `logic signed [31:0]` to support 32-bit accumulation.

3.  **`pe.sv`:**
    *   Accept `sys_mode` as an input.
    *   Internally, widen the accumulator (`accumulator` variable) to `logic signed [31:0]`.
    *   Implement a **Multi-Modal ALU** using a `case (sys_mode)` statement. This ALU will:
        *   Extract appropriate bit-widths (e.g., two 8-bit values from a 16-bit word for INT8 Packed).
        *   Perform multiplications (possibly with sign-extension for lower-bit weights).
        *   Handle fixed-point shifts for Q8.8 mode.
        *   Feed results into the 32-bit accumulator.

4.  **`vpu.sv`:**
    *   Accept `sys_mode` as an input.
    *   Adapt all inputs from the Systolic Array (e.g., `vpu_data_in_1`, `vpu_data_in_2`) to accept `logic signed [31:0]` values.
    *   Introduce a **Requantization Stage** early in the pipeline (e.g., after Bias addition) that uses `sys_mode` to:
        *   Read parameters: `requant_scale`, `requant_shift`, `requant_zero_point` (from UB or separate configuration registers).
        *   Scale and shift the 32-bit accumulated values (e.g., `(acc * scale) >>> shift`).
        *   Perform saturation/clipping to the target output precision (e.g., 16-bit for INT16, 8-bit for INT8, etc.).
        *   This stage will output the lower-precision values (e.g., 16-bit) to the rest of the VPU pipeline (Leaky ReLU, Loss, etc.) and ultimately to the Unified Buffer.

5.  **Unified Buffer:**
    *   The UB will primarily store and retrieve 16-bit data. Its read/write mechanisms will need to understand the `sys_mode` to correctly pack/unpack lower-precision data (e.g., provide two 8-bit values from a single 16-bit address for W8A8 mode).

This comprehensive refactoring will transform the TinyTPU into a far more powerful and versatile accelerator capable of handling diverse quantization schemes efficiently.
