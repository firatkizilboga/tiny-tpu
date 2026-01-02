import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, ReadOnly
import numpy as np

"""
INT8 Packed E2E Test for TinyTPU.

This test mimics test_tpu.py but uses INT8 packed mode (sys_mode=2).
Each 16-bit word holds two INT8 values: (hi, lo).
The PE computes: (input_hi * weight_hi) + (input_lo * weight_lo)
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

def swar_dot(a_packed, w_packed):
    """Compute SWAR dot product: (a_hi * w_hi) + (a_lo * w_lo)."""
    a_hi, a_lo = unpack_int8(a_packed)
    w_hi, w_lo = unpack_int8(w_packed)
    return (a_hi * w_hi) + (a_lo * w_lo)

def compute_expected_outputs(A, W):
    num_input_rows = len(A)
    results = []
    
    for t in range(num_input_rows):
        # Column 0: pe11 then pe21 (psum flows down)
        # Row 0 input is A[t][0], row 1 input is A[t][1]
        # Column 0 weight (after transpose) = W^T col 0 = [W[0][0], W[0][1]]
        pe11_out = swar_dot(A[t][0], W[0][0])  # pe11 weight = W[0][0]
        pe21_out = swar_dot(A[t][1], W[0][1]) + pe11_out  # pe21 weight = W[0][1] (transposed!)
        
        # Column 1: pe12 then pe22 (psum flows down)
        # Column 1 weight (after transpose) = W^T col 1 = [W[1][0], W[1][1]]
        pe12_out = swar_dot(A[t][0], W[1][0])  # pe12 weight = W[1][0] (transposed!)
        pe22_out = swar_dot(A[t][1], W[1][1]) + pe12_out  # pe22 weight = W[1][1]
        
        # Output comes from bottom row: pe21=col0, pe22=col1
        results.append((pe21_out, pe22_out))
    
    return results

def generate_random_packed_matrix(rows, cols, range_min=-128, range_max=127, seed=None):
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

def read_packed_valid(signal, idx):
    try:
        full_val = signal.value.integer
    except ValueError:
        return 0
    return (full_val >> idx) & 1

def read_packed_signed(signal, idx, width=16):
    try:
        full_val = signal.value.integer
    except ValueError:
        return 0
    mask = (1 << width) - 1
    val = (full_val >> (idx * width)) & mask
    if val >= (1 << (width - 1)):
        val -= (1 << width)
    return val

@cocotb.test()
async def test_int8_e2e(dut):
    """INT8 Packed Mode E2E Test"""
    
    DEBUG_MODE = False
    NUM_INPUT_ROWS = 4 if DEBUG_MODE else 50
    SEED = 42
    
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst.value = 1
    dut.sys_mode.value = 2  # INT8 Packed Mode
    
    # Initialize UB signals
    for i in range(2):
        dut.ub_wr_host_data_in[i].value = 0
        dut.ub_wr_host_valid_in[i].value = 0
        
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
    
    if DEBUG_MODE:
        A = [[pack_int8(0, 0)]*2 for _ in range(NUM_INPUT_ROWS)]
        W = [[pack_int8(0, 0)]*2 for _ in range(2)]
    else:
        A = generate_random_packed_matrix(NUM_INPUT_ROWS, 2, range_min=-128, range_max=127, seed=SEED)
        W = generate_random_packed_matrix(2, 2, range_min=-128, range_max=127, seed=SEED + 1)

    # Load A into UB
    dut._log.info(f"Loading {len(A)} input rows to UB...")
    for i in range(len(A)):
        dut.ub_wr_host_data_in[0].value = A[i][0]
        dut.ub_wr_host_valid_in[0].value = 1
        dut.ub_wr_host_data_in[1].value = A[i][1]
        dut.ub_wr_host_valid_in[1].value = 1
        await RisingEdge(dut.clk)
        
    dut.ub_wr_host_valid_in[0].value = 0
    dut.ub_wr_host_valid_in[1].value = 0
    
    w_start_addr = len(A) * 2

    # Load W into UB
    dut._log.info(f"Loading weights to UB at addr {w_start_addr}...")
    for i in range(2):
        dut.ub_wr_host_data_in[0].value = W[i][0]
        dut.ub_wr_host_valid_in[0].value = 1
        dut.ub_wr_host_data_in[1].value = W[i][1]
        dut.ub_wr_host_valid_in[1].value = 1
        await RisingEdge(dut.clk)
    
    dut.ub_wr_host_valid_in[0].value = 0
    dut.ub_wr_host_valid_in[1].value = 0
    await RisingEdge(dut.clk)
    
    # Load W^T into systolic array
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 1
    dut.ub_ptr_select.value = 1
    dut.ub_rd_addr_in.value = w_start_addr
    dut.ub_rd_row_size.value = 2
    dut.ub_rd_col_size.value = 2
    await RisingEdge(dut.clk)
    
    dut.ub_rd_start_in.value = 0
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    await RisingEdge(dut.clk)
    
    # Monitor setup
    captured_1 = []
    captured_2 = []
    
    async def monitor_outputs():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            
            valid_val = 0
            try:
                valid_val = dut.vpu_valid_out.value.integer
            except ValueError:
                valid_val = 0
                
            if (valid_val >> 0) & 1:
                # Capture Channel 1 (Index 0) - Access VPU output directly
                # vpu_data_out is the output signal from VPU
                val = read_packed_signed(dut.vpu_inst.vpu_data_out, 0, 16)
                captured_1.append(val)
                
            if (valid_val >> 1) & 1:
                # Channel 2 (Index 1)
                val = read_packed_signed(dut.vpu_inst.vpu_data_out, 1, 16)
                captured_2.append(val)
    
    monitor_task = cocotb.start_soon(monitor_outputs())
    
    # Stream A into systolic
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = len(A)
    dut.ub_rd_col_size.value = 2
    dut.vpu_data_pathway.value = 0b0000 
    await RisingEdge(dut.clk)
    
    dut.ub_rd_start_in.value = 0
    dut.sys_switch_in.value = 1
    await RisingEdge(dut.clk)
    dut.sys_switch_in.value = 0
    
    # Wait for results
    for _ in range(NUM_INPUT_ROWS + 20):
        await RisingEdge(dut.clk)
        
    monitor_task.kill()
    
    dut._log.info("Computing expected results...")
    expected_results = compute_expected_outputs(A, W)
    expected_col_0 = [res[0] for res in expected_results]
    expected_col_1 = [res[1] for res in expected_results]
    
    assert len(captured_1) >= NUM_INPUT_ROWS, f"Only captured {len(captured_1)} rows on Ch1"
    assert len(captured_2) >= NUM_INPUT_ROWS, f"Only captured {len(captured_2)} rows on Ch2"
    
    valid_captured_1 = captured_1[:NUM_INPUT_ROWS]
    valid_captured_2 = captured_2[:NUM_INPUT_ROWS]
    
    dut._log.info(f"Captured 1: {valid_captured_1[:5]}")
    dut._log.info(f"Expected 1: {expected_col_0[:5]}")
    
    mismatches_col0 = []
    mismatches_col1 = []
    
    for i, (hw, gold) in enumerate(zip(valid_captured_1, expected_col_0)):
        if hw != gold:
            mismatches_col0.append({'row': i, 'hw': hw, 'gold': gold})
            
    for i, (hw, gold) in enumerate(zip(valid_captured_2, expected_col_1)):
        if hw != gold:
            mismatches_col1.append({'row': i, 'hw': hw, 'gold': gold})
            
    if mismatches_col0 or mismatches_col1:
        assert False, f"Verification Failed! {len(mismatches_col0)} Col0 errors, {len(mismatches_col1)} Col1 errors"
        
    dut._log.info(f"✅ INT8 E2E Test PASSED!")
