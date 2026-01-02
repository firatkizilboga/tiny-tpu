# import cocotb
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles
from test_utils import PackedArrayDriver, to_fixed

X = [
    [2., 2.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
]

W1 = [
    [0.2985, -0.5792], 
    [0.0913, 0.4234]
]

# Calculating X @ W1^T
# Expected output:
# [-0.5614  1.0294]
# [-0.5792  0.4234]
# [ 0.2985  0.0913]
# [-0.2807  0.5147]


# First column of accept weight signal turns off -> set switch flag on and set first row start signal on (start loading in X)
@cocotb.test()
async def test_systolic_array(dut): 

    # Create a clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Drivers
    sys_data_drv = PackedArrayDriver(dut.sys_data_in, 16, 2)
    sys_valid_drv = PackedArrayDriver(dut.sys_valid_in, 1, 2)
    sys_switch_drv = PackedArrayDriver(dut.sys_switch_in, 1, 2) # New packed switch
    
    # rst the DUT (device under test)
    dut.rst.value = 1
    dut.sys_accept_w[0].value = 0
    dut.sys_accept_w[1].value = 0
    sys_switch_drv.set_all([0, 0])
    sys_data_drv.set_all([0, 0])
    sys_valid_drv.set_all([0, 0])
    dut.sys_mode.value = 0  # Q8.8 mode
    
    # PE Enable - needs col size valid
    dut.ub_rd_col_size_in.value = 2
    dut.ub_rd_col_size_valid_in.value = 0
    
    await RisingEdge(dut.clk)
    
    # Enable PEs
    dut.rst.value = 0
    dut.ub_rd_col_size_valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.ub_rd_col_size_valid_in.value = 0
    
    # load in transposed weight matrix:
    # Weights for PE column 0 (Row 0 of W1^T -> Col 0 of W1)
    # Weights for PE column 1 (Row 1 of W1^T -> Col 1 of W1)
    
    # W1 = [[W00, W01], [W10, W11]]
    # PE Grid: 2x2.
    # Col 0 needs W[0][0] then W[1][0] (Wait, weights flow top down).
    # If we want W[0][0] at Top, W[1][0] at Bottom?
    # No, we compute X @ W1^T.
    # W1^T = [[W00, W10], [W01, W11]]
    # Col 0 of Systolic corresponds to Col 0 of W1^T?
    # Res[i][0] = X[i] . W1^T[:, 0] = X[i] . [W00, W01]
    # So Col 0 needs W00 and W01.
    # Weights flow Top-Down.
    # Feed W01 then W00.
    
    # Col 1 needs W10 and W11.
    # Feed W11 then W10.
    
    # T1: Feed W01 (Col 0), W11 (Col 1)
    dut.sys_weight_in[0].value = to_fixed(W1[0][1])
    dut.sys_accept_w[0].value = 1
    dut.sys_weight_in[1].value = to_fixed(W1[1][1])
    dut.sys_accept_w[1].value = 1
    await RisingEdge(dut.clk)

    # T2: Feed W00 (Col 0), W10 (Col 1)
    dut.sys_weight_in[0].value = to_fixed(W1[0][0])
    dut.sys_accept_w[0].value = 1
    dut.sys_weight_in[1].value = to_fixed(W1[1][0])
    dut.sys_accept_w[1].value = 1
    await RisingEdge(dut.clk)

    # T3: Switch
    dut.sys_accept_w[0].value = 0
    dut.sys_accept_w[1].value = 0
    sys_switch_drv.set_all([1, 1])
    await RisingEdge(dut.clk)
    
    sys_switch_drv.set_all([0, 0])
    
    # Feed Inputs (Manual Skewing)
    # T4: X00 (Row 0)
    sys_data_drv.set_all([to_fixed(X[0][0]), 0])
    sys_valid_drv.set_all([1, 0])
    await RisingEdge(dut.clk)

    # T5: X10 (Row 0), X01 (Row 1) -> Wait, X01?
    # Input Vector X[0] = [X00, X01].
    # Row 0 gets X00 then X01.
    # Row 1 gets nothing (Single vector mode) OR
    # Is it computing X @ W?
    # X is 4x2. W is 2x2.
    # We feed X rows into Systolic Row 0? 
    # Or X columns into Systolic Rows?
    # 2 inputs to systolic array (Row 0, Row 1).
    # X[t] = [x0, x1].
    # Row 0 gets x0. Row 1 gets x1.
    # Skewing: Row 0 gets x0 at T. Row 1 gets x1 at T+1.
    
    # T5: X[0][0] is now at PE[0][1].
    # We feed X[0][0] at T4.
    # We feed X[0][1] at T5 (Row 1 input).
    sys_data_drv.set_all([to_fixed(X[1][0]), to_fixed(X[0][1])])
    sys_valid_drv.set_all([1, 1])
    await RisingEdge(dut.clk)

    # T6: X[1][0] is at PE[0][0].
    # Feed X[2][0] (Row 0), X[1][1] (Row 1).
    sys_data_drv.set_all([to_fixed(X[2][0]), to_fixed(X[1][1])])
    sys_valid_drv.set_all([1, 1])
    await RisingEdge(dut.clk)

    # T7: Feed X[3][0] (Row 0), X[2][1] (Row 1).
    sys_data_drv.set_all([to_fixed(X[3][0]), to_fixed(X[2][1])])
    sys_valid_drv.set_all([1, 1])
    await RisingEdge(dut.clk)
    
    # T8: Feed 0 (Row 0), X[3][1] (Row 1).
    sys_data_drv.set_all([0, to_fixed(X[3][1])])
    sys_valid_drv.set_all([0, 1])
    await RisingEdge(dut.clk)

    # End
    sys_data_drv.set_all([0, 0])
    sys_valid_drv.set_all([0, 0])
    await RisingEdge(dut.clk)
    
    await ClockCycles(dut.clk, 10)