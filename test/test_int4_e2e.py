import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, ReadOnly
import numpy as np
import os
from test_utils import PackedArrayDriver, read_packed_data

"""
INT4 Packed E2E Test for TinyTPU.

This test uses INT4 packed mode (sys_mode=3).
Each 16-bit word holds four INT4 values: (v3, v2, v1, v0).
The PE computes: dot(input_vec, weight_vec) for 4 elements.

Array size N is configurable via TPU_ARRAY_SIZE environment variable.
"""

# Read array size from environment (set by Makefile)
N = int(os.environ.get('TPU_ARRAY_SIZE', '2'))

def pack_int4(v3, v2, v1, v0):
    """Pack four signed INT4 values into one unsigned 16-bit word."""
    return ((v3 & 0xF) << 12) | ((v2 & 0xF) << 8) | ((v1 & 0xF) << 4) | (v0 & 0xF)

def unpack_int4(packed):
    """Unpack a 16-bit word into four signed INT4 values (v3, v2, v1, v0)."""
    v3 = (packed >> 12) & 0xF
    v2 = (packed >> 8) & 0xF
    v1 = (packed >> 4) & 0xF
    v0 = packed & 0xF
    
    # Convert to signed (4-bit: -8 to +7)
    def to_signed(val):
        return val - 16 if val >= 8 else val
    
    return to_signed(v3), to_signed(v2), to_signed(v1), to_signed(v0)

def swar_dot(a_packed, w_packed):
    """Compute SWAR dot product: 4-way multiply-accumulate."""
    a3, a2, a1, a0 = unpack_int4(a_packed)
    w3, w2, w1, w0 = unpack_int4(w_packed)
    return (a3 * w3) + (a2 * w2) + (a1 * w1) + (a0 * w0)

def compute_expected_outputs(A, W):
    """
    Compute expected systolic array outputs for INT4 packed mode.
    Works for any NxN systolic array.
    A: M x N matrix of packed INT4 values
    W: N x N matrix of packed INT4 values
    Returns: list of N-element tuples (one result per column)
    """
    num_input_rows = len(A)
    num_cols = len(W[0])  # N columns
    results = []
    
    def pe_int4_compute(a_packed, w_packed):
        """Match PE INT4 computation exactly."""
        return swar_dot(a_packed, w_packed)
    
    def requant_16(psum_32):
        """Take low 16 bits as signed (matches VPU requant for mode 2/3)."""
        low16 = psum_32 & 0xFFFF
        if low16 >= 0x8000:
            low16 -= 0x10000
        return low16
    
    for t in range(num_input_rows):
        row_results = []
        for col in range(num_cols):
            # C[t][col] = sum over k of A[t][k] * W[k][col]
            psum = 0
            for k in range(len(W)):  # N rows of W
                psum += pe_int4_compute(A[t][k], W[k][col])
            row_results.append(requant_16(psum))
        results.append(tuple(row_results))
    
    return results


def generate_random_packed_matrix(rows, cols, range_min=-8, range_max=7, seed=None):
    """Generate a random matrix of packed INT4 values."""
    if seed is not None:
        np.random.seed(seed)
    
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            v3 = np.random.randint(range_min, range_max + 1)
            v2 = np.random.randint(range_min, range_max + 1)
            v1 = np.random.randint(range_min, range_max + 1)
            v0 = np.random.randint(range_min, range_max + 1)
            row.append(pack_int4(v3, v2, v1, v0))
        matrix.append(row)
    return matrix


async def wait_for_outputs(dut, num_expected, extra_cycles=20):
    """Wait for all expected outputs to arrive, plus some extra cycles for pipeline flush."""
    # Wait for outputs: input rows + pipeline depth + extra margin
    await ClockCycles(dut.clk, num_expected + extra_cycles)


@cocotb.test()
async def test_int4_e2e(dut):
    """INT4 Packed Mode E2E Test - Full Scale Random Test for NxN Array"""
    
    # Configuration
    NUM_INPUT_ROWS = 50  # Number of rows to test
    SEED = 42
    
    dut._log.info(f"=== INT4 E2E Test with N={N} ===")
    
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize packed array drivers for N lanes
    ub_wr_host_data_drv = PackedArrayDriver(dut.ub_wr_host_data_in, 16, N)
    ub_wr_host_valid_drv = PackedArrayDriver(dut.ub_wr_host_valid_in, 1, N)
    
    # Reset
    dut.rst.value = 1
    dut.sys_mode.value = 3  # INT4 Packed Mode!
    ub_wr_host_data_drv.set_all([0] * N)
    ub_wr_host_valid_drv.set_all([0] * N)
    dut.ub_rd_start_in.value = 0
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    dut.learning_rate_in.value = 0
    dut.vpu_data_pathway.value = 0
    dut.sys_switch_in.value = 0
    dut.vpu_leak_factor_in.value = 0
    dut.inv_batch_size_times_two_in.value = 0
    await RisingEdge(dut.clk)
    
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    # Define test data (Full Range Random Test)
    # A is NUM_INPUT_ROWS x N, W is N x N
    A = generate_random_packed_matrix(NUM_INPUT_ROWS, N, range_min=-8, range_max=7, seed=SEED)
    W = generate_random_packed_matrix(N, N, range_min=-8, range_max=7, seed=SEED + 1)
        
    # Log the generated data
    dut._log.info(f"Input A ({len(A)} rows x {N} cols):")
    for i, row in enumerate(A[:4]):  # Show first 4 rows
        row_str = ", ".join([str(unpack_int4(v)) for v in row])
        dut._log.info(f"  A[{i}] = [{row_str}]")
    if len(A) > 4:
        dut._log.info(f"  ... ({len(A) - 4} more rows)")
    
    dut._log.info(f"Weight W ({N}x{N}):")
    for i, row in enumerate(W):
        row_str = ", ".join([str(unpack_int4(v)) for v in row])
        dut._log.info(f"  W[{i}] = [{row_str}]")
    
    # Load A into UB (wide memory - all columns at once)
    dut._log.info(f"Loading {len(A)} input rows to UB...")
    
    for i, row in enumerate(A):
        ub_wr_host_data_drv.set_all(row)
        ub_wr_host_valid_drv.set_all([1] * N)
        await RisingEdge(dut.clk)
    
    ub_wr_host_valid_drv.set_all([0] * N)
    
    # Calculate where W starts in UB (wide memory: 1 address per row)
    w_start_addr = len(A)

    
    # Load W into UB (wide memory format)
    # REVERSE W so that weights load bottom-up (first in -> bottom row, last in -> top row)
    dut._log.info(f"Loading weights to UB at addr {w_start_addr} (REVERSED ROW ORDER)...")
    W_reversed = W[::-1]
    for i, row in enumerate(W_reversed):
        orig_idx = len(W) - 1 - i 
        dut._log.info(f"Writing W[{orig_idx}] (Rev[{i}])")
        ub_wr_host_data_drv.set_all(row)
        ub_wr_host_valid_drv.set_all([1] * N)
        await RisingEdge(dut.clk)
        
        # Pulse valid low to ensure clean write cycles
        ub_wr_host_valid_drv.set_all([0] * N)
        await RisingEdge(dut.clk)
    
    ub_wr_host_valid_drv.set_all([0] * N)
    await RisingEdge(dut.clk)
    
    # Load W into systolic array (Software transpose/reverse done during loading)
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 0 # Ignored by hardware, but 0 is safe default
    dut.ub_ptr_select.value = 1
    dut.ub_rd_addr_in.value = w_start_addr  # W starts after A
    dut.ub_rd_row_size.value = N  # Number of weight rows
    dut.ub_rd_col_size.value = N  # Number of rows of weights to read (for PE enable)
    await RisingEdge(dut.clk)
    
    dut.ub_rd_start_in.value = 0
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    
    # Wait for weights to propagate through all N rows of the systolic array
    # UB streams N weights over N cycles, then they need N-1 more cycles to reach bottom row
    await ClockCycles(dut.clk, 2 * N - 1)
    
    # Stream A into systolic, switch weights
    
    # Setup monitor to capture outputs for all N channels
    captured = [[] for _ in range(N)]
    
    async def monitor_outputs():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            try:
                valid_val = int(dut.vpu_inst.vpu_valid_out.value)
            except ValueError:
                valid_val = 0
            
            for ch in range(N):
                if (valid_val >> ch) & 1:
                    val = read_packed_data(dut.vpu_inst.vpu_data_out, ch, 16, signed=True)
                    captured[ch].append(val)
    
    monitor_task = cocotb.start_soon(monitor_outputs())
    
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = len(A)  # Number of input rows
    dut.ub_rd_col_size.value = len(A)  # Number of values to read (count)
    dut.vpu_data_pathway.value = 0b0000  # Pass-through (no bias/activation)
    await RisingEdge(dut.clk)
    
    dut.ub_rd_start_in.value = 0
    dut.sys_switch_in.value = 1
    await RisingEdge(dut.clk)
    dut.sys_switch_in.value = 0
    
    # Wait for outputs (need more cycles for larger arrays due to skewing)
    await wait_for_outputs(dut, NUM_INPUT_ROWS, extra_cycles=20 + N * 2)
    monitor_task.cancel()
    
    # Compute Golden Reference
    dut._log.info("Computing expected results using Python golden model...")
    expected_results = compute_expected_outputs(A, W)
    
    # Separate the list of tuples into per-column lists
    expected_cols = [[res[col] for res in expected_results] for col in range(N)]
    
    dut._log.info(f"Golden Col 0 (first 5): {expected_cols[0][:5]}...")
    
    # Verify Results
    for ch in range(N):
        dut._log.info(f"Captured Channel {ch} ({len(captured[ch])} values): {captured[ch][:5]}...")
    
    # Verify we got outputs on all channels
    for ch in range(N):
        assert len(captured[ch]) > 0, f"No outputs captured on channel {ch}!"
    
    # Slice captured results to match input length
    valid_captured = [ch_data[:NUM_INPUT_ROWS] for ch_data in captured]
    
    for ch in range(N):
        assert len(valid_captured[ch]) == NUM_INPUT_ROWS, \
            f"Channel {ch} output length mismatch! Expected {NUM_INPUT_ROWS}, got {len(valid_captured[ch])}"
    
    # Bit-Perfect Verification
    total_mismatches = 0
    mismatches_by_col = []
    
    for col in range(N):
        col_mismatches = []
        dut._log.info(f"Verifying Column {col} against golden model...")
        for i, (hw, gold) in enumerate(zip(valid_captured[col], expected_cols[col])):
            if hw != gold:
                col_mismatches.append({'row': i, 'hw': hw, 'gold': gold})
        mismatches_by_col.append(col_mismatches)
        total_mismatches += len(col_mismatches)
    
    # Report results
    if total_mismatches > 0:
        dut._log.warning(f"Found {total_mismatches} total mismatches across {N} columns")
        for col, mismatches in enumerate(mismatches_by_col):
            if mismatches:
                dut._log.warning(f"  Col {col}: {len(mismatches)} mismatches")
                for m in mismatches[:3]:  # Show first 3
                    dut._log.warning(f"    Row {m['row']}: HW={m['hw']} != Gold={m['gold']}")
        
        # FAIL HARD on any mismatch
        assert False, f"BIT-PERFECT VERIFICATION FAILED! {total_mismatches} total errors"
    else:
        dut._log.info(f"✅ INT4 E2E Test PASSED - BIT-PERFECT MATCH!")
    
    dut._log.info(f"  Array Size: N={N}")
    dut._log.info(f"  Matrix size: {NUM_INPUT_ROWS}x{N} inputs, {N}x{N} weights")
    for ch in range(N):
        dut._log.info(f"  Channel {ch}: {len(captured[ch])} outputs, {len(set(captured[ch]))} unique")