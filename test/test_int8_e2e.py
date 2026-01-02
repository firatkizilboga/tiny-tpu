import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
import numpy as np

"""
INT8 Packed E2E Test for TinyTPU.

This test mimics test_tpu.py but uses INT8 packed mode (sys_mode=2).
Each 16-bit word holds two INT8 values: (hi, lo).
The PE computes: (input_hi * weight_hi) + (input_lo * weight_lo)

We do a simple 4x2 input * 2x2 weight matrix multiply.
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
    """
    Compute expected systolic array outputs for INT8 packed mode.
    
    Args:
        A: Input matrix (rows x cols) of packed uint16 values
        W: Weight matrix (rows x cols) of packed uint16 values
           NOTE: The test loads W with transpose=1, so we need W^T here!
    
    Returns:
        List of (col0_result, col1_result) tuples for each output cycle
    
    Systolic array layout (2x2):
        pe11 (row0,col0) → pe12 (row0,col1)
              ↓                  ↓
        pe21 (row1,col0) → pe22 (row1,col1)
    
    Data flow:
    - sys_data_in_11 feeds pe11 (and flows right to pe12)
    - sys_data_in_21 feeds pe21 (and flows right to pe22)
    - ub_rd_input_data_out_0 → sys_data_in_11 → A[t][0]
    - ub_rd_input_data_out_1 → sys_data_in_21 → A[t][1]
    
    Weight assignment (AFTER transpose load):
    - pe11 gets W[0][0] (column 0, row 0)
    - pe12 gets W[1][0] (was W col 0), flows down from pe11
    - pe21 gets pe_weight_out_11 = W[0][0] → then reloaded? Actually weights FLOW DOWN
    
    Wait - weights flow top-to-bottom within a column:
    - Column 0: pe11 → pe21 (both see weight that came in on sys_weight_in_11)
    - Column 1: pe12 → pe22 (both see weight that came in on sys_weight_in_12)
    
    With transpose, what's on the weight channels over time?
    For a 2x2 matrix, it takes 2 weight load cycles.
    """
    num_input_rows = len(A)
    results = []
    
    # After transpose load, the weight matrix becomes W^T
    # W^T[row][col] = W[col][row]
    # pe11 sees W^T[0][0] = W[0][0]
    # pe12 sees W^T[0][1] = W[1][0]  <- TRANSPOSED!
    # pe21 sees W^T[1][0] = W[0][1]  <- TRANSPOSED!
    # pe22 sees W^T[1][1] = W[1][1]
    
    # Actually, in a weight-stationary array:
    # - Row 0 PEs (pe11, pe12) see sys_data_in_11 = A[t][0]
    # - Row 1 PEs (pe21, pe22) see sys_data_in_21 = A[t][1]
    # - Each column shares the same weight
    # - PSUMs flow top-to-bottom: pe11→pe21, pe12→pe22
    
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



def generate_random_packed_matrix(rows, cols, seed=None):
    """Generate a random matrix of packed INT8 values."""
    if seed is not None:
        np.random.seed(seed)
    
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            # Use smaller range to avoid overflow in SWAR multiply-accumulate
            # Max value: 15*15 + 15*15 = 450 (fits in 32-bit accum easily)
            hi = np.random.randint(-15, 16)
            lo = np.random.randint(-15, 16)
            row.append(pack_int8(hi, lo))
        matrix.append(row)
    return matrix



@cocotb.test()
async def test_int8_e2e(dut):
    """INT8 Packed Mode E2E Test - Debug with simple known values"""
    
    # Configuration - 50 rows to simulate larger matrix operations
    DEBUG_MODE = False  # Set to True for debugging with simple values
    NUM_INPUT_ROWS = 4 if DEBUG_MODE else 50  # 50x2 input for large-scale test
    SEED = 42
    
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst.value = 1
    dut.sys_mode.value = 2  # INT8 Packed Mode!
    dut.ub_wr_host_data_in[0].value = 0
    dut.ub_wr_host_data_in[1].value = 0
    dut.ub_wr_host_valid_in[0].value = 0
    dut.ub_wr_host_valid_in[1].value = 0
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
    
    # --------------------------------
    # Define Simple Test Data for Debugging
    # --------------------------------
    if DEBUG_MODE:
        # Simple values: pack(hi, lo) where SWAR computes (A_hi*W_hi + A_lo*W_lo)
        # Use pack(1, 0) so result is just A_hi * W_hi (makes tracing easier)
        A = [
            [pack_int8(1, 0), pack_int8(2, 0)],   # Row 0: ch0=(1,0), ch1=(2,0)
            [pack_int8(3, 0), pack_int8(4, 0)],   # Row 1: ch0=(3,0), ch1=(4,0)
            [pack_int8(5, 0), pack_int8(6, 0)],   # Row 2: ch0=(5,0), ch1=(6,0)
            [pack_int8(7, 0), pack_int8(8, 0)],   # Row 3: ch0=(7,0), ch1=(8,0)
        ]
        
        W = [
            [pack_int8(1, 0), pack_int8(1, 0)],   # Row 0: all weights = (1,0)
            [pack_int8(1, 0), pack_int8(1, 0)],   # Row 1: all weights = (1,0)
        ]
        
        dut._log.info("DEBUG MODE: Using simple test values A_hi * W_hi = A_hi * 1 = A_hi")
    else:
        A = generate_random_packed_matrix(NUM_INPUT_ROWS, 2, seed=SEED)
        W = generate_random_packed_matrix(2, 2, seed=SEED + 1)

    
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
    
    # --------------------------------
    # Load A into UB
    # --------------------------------
    # Load row by row, channel 0 first then channel 1
    dut._log.info(f"Loading {len(A)} input rows to UB...")
    
    dut.ub_wr_host_data_in[0].value = A[0][0]
    dut.ub_wr_host_valid_in[0].value = 1
    await RisingEdge(dut.clk)
    
    for i in range(len(A) - 1):
        dut.ub_wr_host_data_in[0].value = A[i+1][0]
        dut.ub_wr_host_valid_in[0].value = 1
        dut.ub_wr_host_data_in[1].value = A[i][1]
        dut.ub_wr_host_valid_in[1].value = 1
        await RisingEdge(dut.clk)
    
    # Last column value (use last index, not hardcoded 3)
    dut.ub_wr_host_data_in[0].value = 0
    dut.ub_wr_host_valid_in[0].value = 0
    dut.ub_wr_host_data_in[1].value = A[len(A)-1][1]
    dut.ub_wr_host_valid_in[1].value = 1
    await RisingEdge(dut.clk)
    
    dut.ub_wr_host_valid_in[1].value = 0
    
    # Calculate where W starts in UB
    w_start_addr = len(A) * 2  # Each row takes 2 addresses (2 columns)

    
    # --------------------------------
    # Load W into UB (starting at w_start_addr)
    # --------------------------------
    dut._log.info(f"Loading weights to UB at addr {w_start_addr}...")
    dut.ub_wr_host_data_in[0].value = W[0][0]
    dut.ub_wr_host_valid_in[0].value = 1
    await RisingEdge(dut.clk)
    
    dut.ub_wr_host_data_in[0].value = W[1][0]
    dut.ub_wr_host_valid_in[0].value = 1
    dut.ub_wr_host_data_in[1].value = W[0][1]
    dut.ub_wr_host_valid_in[1].value = 1
    await RisingEdge(dut.clk)
    
    dut.ub_wr_host_data_in[0].value = 0
    dut.ub_wr_host_valid_in[0].value = 0
    dut.ub_wr_host_data_in[1].value = W[1][1]
    dut.ub_wr_host_valid_in[1].value = 1
    await RisingEdge(dut.clk)
    
    dut.ub_wr_host_valid_in[1].value = 0
    await RisingEdge(dut.clk)
    
    # --------------------------------
    # Load W^T into systolic array
    # --------------------------------
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 1
    dut.ub_ptr_select.value = 1
    dut.ub_rd_addr_in.value = w_start_addr  # W starts after A
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
    
    # --------------------------------
    # Stream A into systolic, switch weights
    # --------------------------------
    
    # Setup monitor to capture outputs
    captured_1 = []
    captured_2 = []
    
    async def monitor_outputs():
        while True:
            await RisingEdge(dut.clk)
            if dut.vpu_valid_out_1.value == 1:
                captured_1.append(int(dut.vpu_inst.vpu_data_in_1.value.to_signed()))
            if dut.vpu_valid_out_2.value == 1:
                captured_2.append(int(dut.vpu_inst.vpu_data_in_2.value.to_signed()))
    
    monitor_task = cocotb.start_soon(monitor_outputs())
    
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = len(A)  # Number of input rows
    dut.ub_rd_col_size.value = 2
    dut.vpu_data_pathway.value = 0b0000  # Pass-through (no bias/activation)
    await RisingEdge(dut.clk)
    
    dut.ub_rd_start_in.value = 0
    dut.sys_switch_in.value = 1
    await RisingEdge(dut.clk)
    dut.sys_switch_in.value = 0
    
    # Wait for outputs
    await FallingEdge(dut.vpu_valid_out_1)
    # --------------------------------
    # Compute Golden Reference
    # --------------------------------
    dut._log.info("Computing expected results using Python golden model...")
    expected_results = compute_expected_outputs(A, W)
    
    # Separate the list of tuples [(col0, col1), ...] into two lists
    expected_col_0 = [res[0] for res in expected_results]
    expected_col_1 = [res[1] for res in expected_results]
    
    dut._log.info(f"Golden Col 0 (first 5): {expected_col_0[:5]}...")
    dut._log.info(f"Golden Col 1 (first 5): {expected_col_1[:5]}...")
    
    # --------------------------------
    # Verify Results
    # --------------------------------
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
    
    # --------------------------------
    # Bit-Perfect Verification
    # --------------------------------
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
