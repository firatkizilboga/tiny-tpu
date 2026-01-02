import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, ReadOnly
import numpy as np
from test_utils import PackedArrayDriver, read_packed_data

"""
INT8 Packed E2E Test for TinyTPU.

This test mimics test_tpu.py but uses INT8 packed mode (sys_mode=2).
Each 16-bit word holds two INT8 values: (hi, lo).
The PE computes: (input_hi * weight_hi) + (input_lo * weight_lo)

We do a 50x2 input * 2x2 weight matrix multiply.
"""

def pack_int8(hi, lo):
    """Pack two signed INT8 values into one unsigned 16-bit word."""
    return ((hi & 0xFF) << 8) | (lo & 0xFF)

def unpack_int8(packed):
    """Unpack a 16-bit word into two signed INT8 values (hi, lo)."""
    hi = (packed >> 8) & 0xFF
    lo = packed & 0xFF
    # Convert to signed
    if hi >= 128:
        hi -= 256
    if lo >= 128:
        lo -= 256
    return hi, lo

def compute_expected_outputs(A, W):
    """
    Compute expected systolic array outputs for INT8 packed mode.
    """
    num_input_rows = len(A)
    results = []
    
    def pe_int8_compute(a_packed, w_packed):
        """Match PE INT8 computation exactly."""
        a_hi, a_lo = unpack_int8(a_packed)
        w_hi, w_lo = unpack_int8(w_packed)
        return (a_hi * w_hi) + (a_lo * w_lo)
    
    def requant_16(psum_32):
        """Take low 16 bits as signed (matches VPU requant for mode 2)."""
        low16 = psum_32 & 0xFFFF
        if low16 >= 0x8000:
            low16 -= 0x10000
        return low16
    
    for t in range(num_input_rows):
        # C[t][0] = A[t][0]*W[0][0] + A[t][1]*W[1][0]
        col0 = pe_int8_compute(A[t][0], W[0][0]) + pe_int8_compute(A[t][1], W[1][0])
        
        # C[t][1] = A[t][0]*W[0][1] + A[t][1]*W[1][1]
        col1 = pe_int8_compute(A[t][0], W[0][1]) + pe_int8_compute(A[t][1], W[1][1])
        
        results.append((requant_16(col0), requant_16(col1)))
    
    return results


def generate_random_packed_matrix(rows, cols, range_min=-128, range_max=127, seed=None):
    """Generate a random matrix of packed INT8 values."""
    if seed is not None:
        np.random.seed(seed)
    
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            hi = np.random.randint(range_min, range_max + 1)
            lo = np.random.randint(range_min, range_max + 1)
            row.append(pack_int8(hi, lo))
        matrix.append(row)
    return matrix


async def wait_for_outputs(dut, num_expected, extra_cycles=20):
    """Wait for all expected outputs to arrive, plus some extra cycles for pipeline flush."""
    # Wait for outputs: input rows + pipeline depth + extra margin
    await ClockCycles(dut.clk, num_expected + extra_cycles)


@cocotb.test()
async def test_int8_e2e(dut):
    """INT8 Packed Mode E2E Test - Full Scale Random Test"""
    
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
    dut.sys_mode.value = 2  # INT8 Packed Mode!
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
    A = generate_random_packed_matrix(NUM_INPUT_ROWS, 2, range_min=-128, range_max=127, seed=SEED)
    W = generate_random_packed_matrix(2, 2, range_min=-128, range_max=127, seed=SEED + 1)
        
    # Log the generated data
    dut._log.info(f"Input A ({len(A)} rows):")
    for i, row in enumerate(A[:4]):  # Show first 4 rows
        hi0, lo0 = unpack_int8(row[0])
        hi1, lo1 = unpack_int8(row[1])
        dut._log.info(f"  A[{i}] = [({hi0},{lo0}), ({hi1},{lo1})]")
    if len(A) > 4:
        dut._log.info(f"  ... ({len(A) - 4} more rows)")
    
    dut._log.info(f"Weight W:")
    for i, row in enumerate(W):
        hi0, lo0 = unpack_int8(row[0])
        hi1, lo1 = unpack_int8(row[1])
        dut._log.info(f"  W[{i}] = [({hi0},{lo0}), ({hi1},{lo1})]")
    
    # Load A into UB (wide memory - all columns at once)
    dut._log.info(f"Loading {len(A)} input rows to UB...")
    
    for i, row in enumerate(A):
        # dut._log.info(f"Writing A[{i}]: packed=[0x{row[0]:04x}, 0x{row[1]:04x}]")
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
            hi0, lo0 = unpack_int8(A[i][0])
            hi1, lo1 = unpack_int8(A[i][1])
            mismatches_col0.append({
                'row': i, 'hw': hw, 'gold': gold, 
                'input': f"A[{i}]=({hi0},{lo0}),({hi1},{lo1})"
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
        dut._log.info(f"✅ INT8 E2E Test PASSED - BIT-PERFECT MATCH!")
    
    dut._log.info(f"  Matrix size: {NUM_INPUT_ROWS}x2 inputs, 2x2 random weights")
    dut._log.info(f"  Channel 1: {len(captured_1)} outputs, {len(set(captured_1))} unique")
    dut._log.info(f"  Channel 2: {len(captured_2)} outputs, {len(set(captured_2))} unique")