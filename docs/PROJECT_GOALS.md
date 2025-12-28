# TinyTPU Project Goals & Architectural Vision

## 1. Project Objective
The primary goal of the TinyTPU project is to design, simulate, and implement a **realistic, education-oriented Tensor Processing Unit (TPU)** from scratch using SystemVerilog. Unlike toy models that abstract away complexity, this project aims to replicate the architectural constraints and design patterns found in commercial AI accelerators (like Google's TPU or NVIDIA's NPU cores), tailored for deployment on FPGA.

The project focuses on the "First Principles" of Deep Learning hardware:
1.  **Massive Parallelism:** Utilizing a Systolic Array for dense matrix multiplication.
2.  **Dataflow & Reuse:** Optimizing memory bandwidth via weight-stationary dataflow.
3.  **Quantization:** Implementing hardware support for low-precision arithmetic (INT8/INT4) to maximize throughput and efficiency.

---

## 2. Core Architecture

### A. The Engine: Systolic Array
*   **Topology:** 2D Grid of Processing Elements (PEs).
*   **Dataflow:** Weight-Stationary.
    *   *Weights* are pre-loaded and stationary inside PEs.
    *   *Inputs* flow horizontally (Left to Right).
    *   *Partial Sums* flow vertically (Top to Bottom).
*   **Precision Modes (Multi-Modal PE):**
    The design moves beyond simple fixed-point arithmetic to support modern "Packed" execution:
    *   **Q8.8 Fixed-Point (Baseline):** High-precision mode (16-bit).
    *   **W8A8 (INT8 Packed):** 2x Throughput. Packs two 8-bit operations into a single 16-bit word.
    *   **W4A4 (INT4 Packed):** 4x Throughput. Packs four 4-bit operations into a single 16-bit word.
    *   **Hybrid Modes (W4A8):** Leveraging bandwidth asymmetry to reduce memory pressure (Memory Reuse).

### B. The Nexus: Unified Buffer (UB)
*   **Role:** Central Scratchpad Memory.
*   **Constraint:** strictly 16-bit architecture.
*   **Function:** Handles all IO, storing Weights, Activations, Biases, and Gradients indiscriminately. It acts as the bridge between the slow host (CPU/RAM) and the fast Systolic Array.

### C. The Post-Processor: Vector Processing Unit (VPU)
*   **Role:** Element-wise operations pipeline.
*   **Pipeline Stages:**
    1.  **Bias Addition:** Adds learned bias terms to matrix results.
    2.  **Requantization:** Scales and shifts wide 32-bit accumulators back down to 16-bit/8-bit storage formats.
    3.  **Activation:** Non-linear functions (ReLU, Leaky ReLU).
    4.  **Training Support:** Gradient calculation (Loss derivatives) and backward pass logic.

---

## 3. Key Technical Challenges & "The Gap"

This project specifically targets the gap between "Hardware Logic" and "Deep Learning Theory."

*   **Precision Management:** Handling the transition from 32-bit internal accumulators (necessary to prevent overflow) back to 16-bit storage (necessary for bandwidth).
*   **Packing & Alignment:** Implementing "SWAR" (SIMD Within A Register) logic to process multiple low-precision numbers on a single physical wire.
*   **Timing & Skew:** Managing the complex "wavefront" timing of a Systolic Array, where data must arrive at specific PEs at specific clock cycles (input skewing).

## 4. End Goal Deliverable
A fully synthesizable SystemVerilog IP core that can:
1.  Load a quantized Neural Network model (Weights/Biases).
2.  Execute Inference (Forward Pass) on input images.
3.  Support limited On-Chip Training (Backward Pass/Gradient Descent).
4.  Demonstrate the performance trade-offs between precision (INT16 vs INT8 vs INT4) and speed.
