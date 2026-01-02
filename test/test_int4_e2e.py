import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
import numpy as np

"""
INT4 Packed E2E Test for TinyTPU.

This test mimics test_int8_e2e.py but uses INT4 packed mode (sys_mode=3).
Each 16-bit word holds four INT4 values: (v3, v2, v1, v0).
The PE computes: (input_v3 * weight_v3) + (input_v2 * weight_v2) + 
                 (input_v1 * weight_v1) + (input_v0 * weight_v0)

We do a simple 4x2 input * 2x2 weight matrix multiply.
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
    - Weights flow top-to-bottom within a column
    - Column 0: pe11 → pe21 (both see weight that came in on sys_weight_in_11)
    - Column 1: pe12 → pe22 (both see weight that came in on sys_weight_in_12)
    
    After transpose load, the weight matrix becomes W^T
    """
    num_input_rows = len(A)
    results = []
    
    # After transpose load, the weight matrix becomes W^T
    # W^T[row][col] = W[col][row]
    # pe11 sees W^T[0][0] = W[0][0]
    # pe12 sees W^T[0][1] = W[1][0]  <- TRANSPOSED!
    # pe21 sees W^T[1][0] = W[0][1]  <- TRANSPOSED!
    # pe22 sees W^T[1][1] = W[1][1]
    
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


def generate_random_packed_matrix(rows, cols, range_min=-8, range_max=7, seed=None):
    """Generate a random matrix of packed INT4 values."""
    if seed is not None:
        np.random.seed(seed)
    
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            # Use full range
            v3 = np.random.randint(range_min, range_max + 1)
            v2 = np.random.randint(range_min, range_max + 1)
            v1 = np.random.randint(range_min, range_max + 1)
            v0 = np.random.randint(range_min, range_max + 1)
            row.append(pack_int4(v3, v2, v1, v0))
        matrix.append(row)
    return matrix



@cocotb.test()
async def test_int4_e2e(dut):
    """INT4 Packed Mode E2E Test - Debug with simple known values"""
    
    # Configuration - 50 rows to simulate larger matrix operations
    DEBUG_MODE = False  # Set to True for debugging with simple values
    NUM_INPUT_ROWS = 4 if DEBUG_MODE else 50  # 50x2 input for large-scale test
    SEED = 42
    
    # Create clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst.value = 1
    dut.sys_mode.value = 3  # INT4 Packed Mode!
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
         # Corner Cases for Robustness
        case = 3
        
        if case == 0:
            # 1. Zero Input
            dut._log.info("DEBUG MODE: Testing All Zeros")
            A = [[pack_int4(0, 0, 0, 0)]*2 for _ in range(NUM_INPUT_ROWS)]
            W = [[pack_int4(0, 0, 0, 0)]*2 for _ in range(2)]
        elif case == 1:
            # 2. Max Positive (7)
            dut._log.info("DEBUG MODE: Testing Max Positive (7)")
            A = [[pack_int4(7, 7, 7, 7)]*2 for _ in range(NUM_INPUT_ROWS)]
            W = [[pack_int4(7, 7, 7, 7)]*2 for _ in range(2)]
        elif case == 2:
            # 3. Max Negative (-8)
            dut._log.info("DEBUG MODE: Testing Max Negative (-8)")
            A = [[pack_int4(-8, -8, -8, -8)]*2 for _ in range(NUM_INPUT_ROWS)]
            W = [[pack_int4(-8, -8, -8, -8)]*2 for _ in range(2)]
        else:
            # 4. Simple arithmetic check
            # Simple values: pack(v3, v2, v1, v0) where SWAR computes sum of 4 products
            # Use pack(1, 1, 1, 1) so result is just sum of inputs (makes tracing easier)
            A = [
                [pack_int4(1, 1, 1, 1), pack_int4(2, 2, 2, 2)],   # Row 0
                [pack_int4(1, 1, 1, 1), pack_int4(2, 2, 2, 2)],   # Row 1
                [pack_int4(1, 1, 1, 1), pack_int4(2, 2, 2, 2)],   # Row 2
                [pack_int4(1, 1, 1, 1), pack_int4(2, 2, 2, 2)],   # Row 3
            ]
            
            W = [
                [pack_int4(1, 1, 1, 1), pack_int4(1, 1, 1, 1)],   # Row 0
                [pack_int4(1, 1, 1, 1), pack_int4(1, 1, 1, 1)],   # Row 1
            ]
            dut._log.info("DEBUG MODE: Using simple test values")
    else:
        # Full Range Random Test (-8 to 7)
        A = generate_random_packed_matrix(NUM_INPUT_ROWS, 2, range_min=-8, range_max=7, seed=SEED)
        W = generate_random_packed_matrix(2, 2, range_min=-8, range_max=7, seed=SEED + 1)

    
    # Log the generated data
    dut._log.info(f"Input A ({len(A)} rows):")
    for i, row in enumerate(A[:4]):  # Show first 4 rows
        v3_0, v2_0, v1_0, v0_0 = unpack_int4(row[0])
        v3_1, v2_1, v1_1, v0_1 = unpack_int4(row[1])
        dut._log.info(f"  A[{i}] = [({v3_0},{v2_0},{v1_0},{v0_0}), ({v3_1},{v2_1},{v1_1},{v0_1})]")
    if len(A) > 4:
        dut._log.info(f"  ... ({len(A) - 4} more rows)")
    
    dut._log.info(f"Weight W:")
    for i, row in enumerate(W):
        v3_0, v2_0, v1_0, v0_0 = unpack_int4(row[0])
        v3_1, v2_1, v1_1, v0_1 = unpack_int4(row[1])
        dut._log.info(f"  W[{i}] = [({v3_0},{v2_0},{v1_0},{v0_0}), ({v3_1},{v2_1},{v1_1},{v0_1})]")
    
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
    
    # Last column value
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
            v3_0, v2_0, v1_0, v0_0 = unpack_int4(A[i][0])
            v3_1, v2_1, v1_1, v0_1 = unpack_int4(A[i][1])
            mismatches_col0.append({
                'row': i, 'hw': hw, 'gold': gold, 
                'input': f"A[{i}]=({v3_0},{v2_0},{v1_0},{v0_0}),({v3_1},{v2_1},{v1_1},{v0_1})"
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
