import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, ReadOnly
import numpy as np
from test_utils import PackedArrayDriver, read_packed_data

"""
INT4 Packed E2E Test for TinyTPU.

This test mimics test_tpu.py but uses INT4 packed mode (sys_mode=3).
Each 16-bit word holds four INT4 values: (v3, v2, v1, v0).
The PE computes: dot(input_vec, weight_vec) for 4 elements.

We do a 50x2 input * 2x2 weight matrix multiply.
"""

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
    """
    num_input_rows = len(A)
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
        # C[t][0] = A[t][0]*W[0][0] + A[t][1]*W[1][0]
        col0 = pe_int4_compute(A[t][0], W[0][0]) + pe_int4_compute(A[t][1], W[1][0])
        
        # C[t][1] = A[t][0]*W[0][1] + A[t][1]*W[1][1]
        col1 = pe_int4_compute(A[t][0], W[0][1]) + pe_int4_compute(A[t][1], W[1][1])
        
        results.append((requant_16(col0), requant_16(col1)))
    
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
    """INT4 Packed Mode E2E Test - Full Scale Random Test"""
    
    # Configuration - 50 rows to simulate larger matrix operations
    NUM_INPUT_ROWS = 50
    SEED = 42
    
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize packed array drivers
    ub_wr_host_data_drv = PackedArrayDriver(dut.ub_wr_host_data_in, 16, 2)
    ub_wr_host_valid_drv = PackedArrayDriver(dut.ub_wr_host_valid_in, 1, 2)
    
    # Reset
    dut.rst.value = 1
    dut.sys_mode.value = 3  # INT4 Packed Mode! (3)
    ub_wr_host_data_drv.set_all([0, 0])
    ub_wr_host_valid_drv.set_all([0, 0])
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
    A = generate_random_packed_matrix(NUM_INPUT_ROWS, 2, range_min=-8, range_max=7, seed=SEED)
    W = generate_random_packed_matrix(2, 2, range_min=-8, range_max=7, seed=SEED + 1)
        
    # Log the generated data
    dut._log.info(f"Input A ({len(A)} rows):")
    for i, row in enumerate(A[:4]):  # Show first 4 rows
        v3_0, v2_0, v1_0, v0_0 = unpack_int4(row[0])
        v3_1, v2_1, v1_1, v0_1 = unpack_int4(row[1])
        dut._log.info(f"  A[{i}] = [({v3_0},{v2_0},{v1_0},{v0_0}), ({v3_1},{v2_1},{v1_1},{v0_1})")
    if len(A) > 4:
        dut._log.info(f"  ... ({len(A) - 4} more rows)")
    
    dut._log.info(f"Weight W:")
    for i, row in enumerate(W):
        v3_0, v2_0, v1_0, v0_0 = unpack_int4(row[0])
        v3_1, v2_1, v1_1, v0_1 = unpack_int4(row[1])
        dut._log.info(f"  W[{i}] = [({v3_0},{v2_0},{v1_0},{v0_0}), ({v3_1},{v2_1},{v1_1},{v0_1})")
    
    # Load A into UB (wide memory - all columns at once)
    dut._log.info(f"Loading {len(A)} input rows to UB...")
    
    for i, row in enumerate(A):
        ub_wr_host_data_drv.set_all([row[0], row[1]])
        ub_wr_host_valid_drv.set_all([1, 1])
        await RisingEdge(dut.clk)
    
    ub_wr_host_valid_drv.set_all([0, 0])
    
    # Calculate where W starts in UB (wide memory: 1 address per row)
    w_start_addr = len(A)

    
    # Load W into UB (wide memory format)
    # REVERSE W so that weights load bottom-up (first in -> bottom row, last in -> top row)
    dut._log.info(f"Loading weights to UB at addr {w_start_addr} (REVERSED ROW ORDER)...")
    W_reversed = W[::-1]
    for i, row in enumerate(W_reversed):
        # We assume W has 2 rows [0, 1]. i=0 writes W[1]. i=1 writes W[0].
        # Log which ORIGINAL row we are writing helps debug
        orig_idx = len(W) - 1 - i 
        dut._log.info(f"Writing W[{orig_idx}] (Rev[{i}]): packed=[0x{row[0]:04x}, 0x{row[1]:04x}]")
        ub_wr_host_data_drv.set_all([row[0], row[1]])
        ub_wr_host_valid_drv.set_all([1, 1])
        await RisingEdge(dut.clk)
        
        # Pulse valid low to ensure clean write cycles
        ub_wr_host_valid_drv.set_all([0, 0])
        await RisingEdge(dut.clk)
    
    ub_wr_host_valid_drv.set_all([0, 0])
    await RisingEdge(dut.clk)
    
    # Load W into systolic array (Software transpose/reverse done during loading)
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 0 # Ignored by hardware, but 0 is safe default
    dut.ub_ptr_select.value = 1
    dut.ub_rd_addr_in.value = w_start_addr  # W starts after A
    dut.ub_rd_row_size.value = len(W) # Ignored by hardware?
    dut.ub_rd_col_size.value = len(W)  # Number of rows of weights to read
    await RisingEdge(dut.clk)
    
    dut.ub_rd_start_in.value = 0
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    await RisingEdge(dut.clk)
    
    # Stream A into systolic, switch weights
    
    # Setup monitor to capture outputs
    captured_1 = []
    captured_2 = []
    
    async def monitor_outputs():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            try:
                valid_val = int(dut.vpu_inst.vpu_valid_out.value)
            except ValueError:
                valid_val = 0
            
            if (valid_val >> 0) & 1:
                val = read_packed_data(dut.vpu_inst.vpu_data_out, 0, 16, signed=True)
                captured_1.append(val)
            if (valid_val >> 1) & 1:
                val = read_packed_data(dut.vpu_inst.vpu_data_out, 1, 16, signed=True)
                captured_2.append(val)
    
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
    
    # Wait for outputs
    await wait_for_outputs(dut, NUM_INPUT_ROWS)
    monitor_task.cancel()
    
    # Compute Golden Reference
    dut._log.info("Computing expected results using Python golden model...")
    expected_results = compute_expected_outputs(A, W)
    
    # Separate the list of tuples [(col0, col1), ...] into two lists
    expected_col_0 = [res[0] for res in expected_results]
    expected_col_1 = [res[1] for res in expected_results]
    
    dut._log.info(f"Golden Col 0 (first 5): {expected_col_0[:5]}...")
    dut._log.info(f"Golden Col 1 (first 5): {expected_col_1[:5]}...")
    
    # Verify Results
    dut._log.info(f"Captured Output 1 ({len(captured_1)} values): {captured_1[:5]}...")
    dut._log.info(f"Captured Output 2 ({len(captured_2)} values): {captured_2[:5]}...")
    
    # Verify we got outputs on both channels
    assert len(captured_1) > 0, "No outputs captured on channel 1!"
    assert len(captured_2) > 0, "No outputs captured on channel 2!"
    
    # Slice captured results to match input length (in case of pipeline latency effects)
    valid_captured_1 = captured_1[:NUM_INPUT_ROWS]
    valid_captured_2 = captured_2[:NUM_INPUT_ROWS]
    
    assert len(valid_captured_1) == NUM_INPUT_ROWS, \
        f"Output length mismatch! Expected {NUM_INPUT_ROWS}, got {len(valid_captured_1)}"
    
    # Bit-Perfect Verification
    mismatches_col0 = []
    mismatches_col1 = []
    
    dut._log.info("Verifying Column 0 (Channel 1) against golden model...")
    for i, (hw, gold) in enumerate(zip(valid_captured_1, expected_col_0)):
        if hw != gold:
            v3_0, v2_0, v1_0, v0_0 = unpack_int4(A[i][0])
            v3_1, v2_1, v1_1, v0_1 = unpack_int4(A[i][1])
            mismatches_col0.append({
                'row': i, 'hw': hw, 'gold': gold, 
                'input': f"A[{i}]=({v3_0},{v2_0},{v1_0},{v0_0}), ({v3_1},{v2_1},{v1_1},{v0_1})"
            })
    
    dut._log.info("Verifying Column 1 (Channel 2) against golden model...")
    for i, (hw, gold) in enumerate(zip(valid_captured_2, expected_col_1)):
        if hw != gold:
            mismatches_col1.append({'row': i, 'hw': hw, 'gold': gold})
    
    # Report results
    if mismatches_col0 or mismatches_col1:
        dut._log.warning(f"Found {len(mismatches_col0)} Col0 mismatches, {len(mismatches_col1)} Col1 mismatches")
        for m in mismatches_col0[:3]:  # Show first 3
            dut._log.warning(f"  Col0 Row {m['row']}: HW={m['hw']} != Gold={m['gold']} ({m['input']})")
        for m in mismatches_col1[:3]:
            dut._log.warning(f"  Col1 Row {m['row']}: HW={m['hw']} != Gold={m['gold']}")
        
        # FAIL HARD on any mismatch - no fallback!
        assert False, f"BIT-PERFECT VERIFICATION FAILED! {len(mismatches_col0)} Col0 errors, {len(mismatches_col1)} Col1 errors"
    else:
        dut._log.info(f"✅ INT4 E2E Test PASSED - BIT-PERFECT MATCH!")
    
    dut._log.info(f"  Matrix size: {NUM_INPUT_ROWS}x2 inputs, 2x2 random weights")
    dut._log.info(f"  Channel 1: {len(captured_1)} outputs, {len(set(captured_1))} unique")
    dut._log.info(f"  Channel 2: {len(captured_2)} outputs, {len(set(captured_2))} unique")