import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, ReadOnly
import numpy as np

"""
INT4 Packed E2E Test for TinyTPU.
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
    
    def to_signed(val):
        return val - 16 if val >= 8 else val
    
    return to_signed(v3), to_signed(v2), to_signed(v1), to_signed(v0)

def swar_dot(a_packed, w_packed):
    """Compute SWAR dot product: 4-way multiply-accumulate."""
    a3, a2, a1, a0 = unpack_int4(a_packed)
    w3, w2, w1, w0 = unpack_int4(w_packed)
    return (a3 * w3) + (a2 * w2) + (a1 * w1) + (a0 * w0)

def compute_expected_outputs(A, W):
    num_input_rows = len(A)
    results = []
    
    for t in range(num_input_rows):
        pe11_out = swar_dot(A[t][0], W[0][0])
        pe21_out = swar_dot(A[t][1], W[0][1]) + pe11_out
        
        pe12_out = swar_dot(A[t][0], W[1][0])
        pe22_out = swar_dot(A[t][1], W[1][1]) + pe12_out
        
        results.append((pe21_out, pe22_out))
    
    return results

def generate_random_packed_matrix(rows, cols, range_min=-8, range_max=7, seed=None):
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
async def test_int4_e2e(dut):
    """INT4 Packed Mode E2E Test"""
    
    DEBUG_MODE = False
    NUM_INPUT_ROWS = 4 if DEBUG_MODE else 50
    SEED = 42
    
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    dut.rst.value = 1
    dut.sys_mode.value = 3  # INT4 Packed Mode!
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
         A = [[pack_int4(1, 1, 1, 1)]*2 for _ in range(NUM_INPUT_ROWS)]
         W = [[pack_int4(1, 1, 1, 1)]*2 for _ in range(2)]
    else:
        A = generate_random_packed_matrix(NUM_INPUT_ROWS, 2, range_min=-8, range_max=7, seed=SEED)
        W = generate_random_packed_matrix(2, 2, range_min=-8, range_max=7, seed=SEED + 1)

    # Load A
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

    # Load W
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
                val = read_packed_signed(dut.vpu_inst.vpu_data_out, 0, 16)
                captured_1.append(val)
                
            if (valid_val >> 1) & 1:
                val = read_packed_signed(dut.vpu_inst.vpu_data_out, 1, 16)
                captured_2.append(val)
    
    monitor_task = cocotb.start_soon(monitor_outputs())
    
    # Stream A
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
    
    assert len(captured_1) >= NUM_INPUT_ROWS, f"Only captured {len(captured_1)} on Ch1"
    assert len(captured_2) >= NUM_INPUT_ROWS, f"Only captured {len(captured_2)} on Ch2"
    
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
        
    dut._log.info(f"✅ INT4 E2E Test PASSED!")
