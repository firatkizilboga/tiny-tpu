import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles
from test_utils import PackedArrayDriver

def to_fixed(val, frac_bits=8):
    """Convert a float to 16-bit fixed point with 8 fractional bits."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF

X = [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
]

@cocotb.test()
async def test_unified_buffer(dut):

    # Create a clock
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Drivers
    ub_wr_drv = PackedArrayDriver(dut.ub_wr_data_in, 16, 2)
    ub_wr_valid_drv = PackedArrayDriver(dut.ub_wr_valid_in, 1, 2)
    ub_wr_host_drv = PackedArrayDriver(dut.ub_wr_host_data_in, 16, 2)
    ub_wr_host_valid_drv = PackedArrayDriver(dut.ub_wr_host_valid_in, 1, 2)

    # rst the DUT (device under test)
    dut.rst.value = 1
    ub_wr_drv.set_all([0, 0])
    ub_wr_valid_drv.set_all([0, 0])
    ub_wr_host_drv.set_all([0, 0])
    ub_wr_host_valid_drv.set_all([0, 0])
    
    await RisingEdge(dut.clk)

    dut.rst.value = 0
    dut.learning_rate_in.value = to_fixed(2)
    
    await RisingEdge(dut.clk)

    # write to UB
    ub_wr_host_drv.set_all([to_fixed(X[0][0]), 0])
    ub_wr_host_valid_drv.set_all([1, 0])
    await RisingEdge(dut.clk)

    ub_wr_host_drv.set_all([to_fixed(X[1][0]), to_fixed(X[0][1])])
    ub_wr_host_valid_drv.set_all([1, 1])
    await RisingEdge(dut.clk)

    ub_wr_host_drv.set_all([to_fixed(X[2][0]), to_fixed(X[1][1])])
    ub_wr_host_valid_drv.set_all([1, 1])
    await RisingEdge(dut.clk)

    ub_wr_host_drv.set_all([to_fixed(X[3][0]), to_fixed(X[2][1])])
    ub_wr_host_valid_drv.set_all([1, 1])
    await RisingEdge(dut.clk)

    ub_wr_host_drv.set_all([0, to_fixed(X[3][1])])
    ub_wr_host_valid_drv.set_all([0, 1])
    await RisingEdge(dut.clk)

    ub_wr_host_drv.set_all([0, 0])
    ub_wr_host_valid_drv.set_all([0, 0])
    await RisingEdge(dut.clk)

    # Reading inputs from UB to left side of systolic array (untransposed)
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 0     # Selecting input pointer
    dut.ub_rd_addr_in.value = 2
    # dut.ub_rd_row_size.value = 3 # REMOVED in UB
    dut.ub_rd_count.value = 3 # Count instead
    # dut.ub_rd_transpose.value = 0 # REMOVED in UB
    await RisingEdge(dut.clk)

    # Reading weights from UB to top of systolic array (untransposed)
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 1
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_count.value = 3
    await RisingEdge(dut.clk)

    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_count.value = 0
    await RisingEdge(dut.clk)

    await ClockCycles(dut.clk, 6)

    # Reading inputs from UB to left side of systolic array (transposed)
    # The new UB logic DOES NOT support hardware transpose.
    # The signals ub_rd_transpose are gone.
    # The tests for transposition will fail if we expect transposition.
    # However, this test is checking 'read' functionality.
    # If the RTL was updated to remove transpose, we should test linear reads.
    
    # dut.ub_rd_start_in.value = 1
    # dut.ub_ptr_select.value = 0
    # dut.ub_rd_addr_in.value = 0
    # dut.ub_rd_count.value = 3
    # await RisingEdge(dut.clk)

    # ... I will comment out the transpose section or adapt it to linear reads
    # For now, let's just do another linear read to check pointers.
    
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_count.value = 3
    await RisingEdge(dut.clk)
    
    dut.ub_rd_start_in.value = 0
    await RisingEdge(dut.clk)

    await ClockCycles(dut.clk, 6)

    # Reading bias from UB to bias modules in VPU
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 2
    dut.ub_rd_addr_in.value = 5
    dut.ub_rd_count.value = 3
    await RisingEdge(dut.clk)

    # Reading Y from UB to loss modules in VPU
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 3
    dut.ub_rd_addr_in.value = 2
    dut.ub_rd_count.value = 2
    await RisingEdge(dut.clk)

    # Reading H from UB to activation derivative modules in VPU
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 4
    dut.ub_rd_addr_in.value = 4
    dut.ub_rd_count.value = 2
    await RisingEdge(dut.clk)

    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_count.value = 0
    await RisingEdge(dut.clk)

    await ClockCycles(dut.clk, 6)

    # Testing gradient descent (biases)
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 5
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_count.value = 2
    await RisingEdge(dut.clk)

    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_count.value = 0
    
    # Simulate VPU sending gradients back
    ub_wr_drv.set_all([to_fixed(X[2][0]), 0])
    ub_wr_valid_drv.set_all([1, 0])
    await RisingEdge(dut.clk)

    ub_wr_drv.set_all([to_fixed(X[3][0]), to_fixed(X[2][1])])
    ub_wr_valid_drv.set_all([1, 1])
    await RisingEdge(dut.clk)

    ub_wr_drv.set_all([0, to_fixed(X[3][1])])
    ub_wr_valid_drv.set_all([0, 1])
    await RisingEdge(dut.clk)

    ub_wr_drv.set_all([0, 0])
    ub_wr_valid_drv.set_all([0, 0])
    await RisingEdge(dut.clk)

    await ClockCycles(dut.clk, 6)

    # Testing gradient descent (weights)
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 6
    dut.ub_rd_addr_in.value = 4
    dut.ub_rd_count.value = 2
    await RisingEdge(dut.clk)

    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_count.value = 0
    
    ub_wr_drv.set_all([to_fixed(X[0][0]), 0])
    ub_wr_valid_drv.set_all([1, 0])
    await RisingEdge(dut.clk)

    ub_wr_drv.set_all([to_fixed(X[1][0]), to_fixed(X[0][1])])
    ub_wr_valid_drv.set_all([1, 1])
    await RisingEdge(dut.clk)

    ub_wr_drv.set_all([0, to_fixed(X[1][1])])
    ub_wr_valid_drv.set_all([0, 1])
    await RisingEdge(dut.clk)

    ub_wr_drv.set_all([0, 0])
    ub_wr_valid_drv.set_all([0, 0])
    await RisingEdge(dut.clk)

    await ClockCycles(dut.clk, 10)
