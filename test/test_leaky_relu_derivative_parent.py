import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer

FRAC_BITS = 8
N_LANES = 2

def to_fixed(val, frac_bits=FRAC_BITS):
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF

def from_fixed(val, frac_bits=FRAC_BITS):
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)

def compute_leaky_relu_derivative(data, leak_factor):
    # This logic assumes H == data for the purpose of the test
    if data >= 0:
        return data
    else:
        return data * leak_factor

class PackedArrayDriver:
    def __init__(self, signal, width, n_lanes=N_LANES):
        self.signal = signal
        self.width = width
        self.n_lanes = n_lanes
        self.values = [0] * n_lanes

    def set(self, idx, val):
        self.values[idx] = val
        self.commit()

    def commit(self):
        full_val = 0
        for i in range(self.n_lanes):
            val = int(self.values[i]) & ((1 << self.width) - 1)
            full_val |= (val << (i * self.width))
        self.signal.value = full_val

def read_packed(signal, idx, width=16):
    try:
        full_val = signal.value.integer
    except ValueError:
        return 0
    mask = (1 << width) - 1
    return (full_val >> (idx * width)) & mask

def read_packed_valid(signal, idx):
    try:
        full_val = signal.value.integer
    except ValueError:
        return 0
    return (full_val >> idx) & 1

async def monitor_outputs(dut, col1_results, col2_results):
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        
        if read_packed_valid(dut.lr_d_valid_out, 0):
            val = read_packed(dut.lr_d_data_out, 0, 16)
            col1_results.append(from_fixed(val))
            
        if read_packed_valid(dut.lr_d_valid_out, 1):
            val = read_packed(dut.lr_d_data_out, 1, 16)
            col2_results.append(from_fixed(val))

BATCH_4x2_DATA_COL1 = [2.5, -1.2, 0.8, -3.1]
BATCH_4x2_DATA_COL2 = [1.8, -0.9, 1.5, -2.2]
LEAK_FACTOR = 0.1

@cocotb.test()
async def test_leaky_relu_derivative_parent_4x2_staggered(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    lr_d_valid_drv = PackedArrayDriver(dut.lr_d_valid_in, 1)
    lr_d_data_drv = PackedArrayDriver(dut.lr_d_data_in, 16)
    lr_d_H_drv = PackedArrayDriver(dut.lr_d_H_in, 16) # Add H driver
    
    dut.rst.value = 1
    lr_d_valid_drv.commit()
    lr_d_data_drv.commit()
    lr_d_H_drv.commit()
    dut.lr_leak_factor_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    
    dut.lr_leak_factor_in.value = to_fixed(LEAK_FACTOR)
    
    col1_results = []
    col2_results = []
    monitor = cocotb.start_soon(monitor_outputs(dut, col1_results, col2_results))
    
    staggered_pattern = [
        (True, False),
        (True, True),
        (True, True),
        (True, True),
        (False, True),
    ]
    
    col1_idx = 0
    col2_idx = 0
    
    for i, (use_col1, use_col2) in enumerate(staggered_pattern):
        if use_col1 and col1_idx < len(BATCH_4x2_DATA_COL1):
            val = BATCH_4x2_DATA_COL1[col1_idx]
            lr_d_data_drv.set(0, to_fixed(val))
            lr_d_H_drv.set(0, to_fixed(val)) # Drive H with same value
            lr_d_valid_drv.set(0, 1)
            col1_idx += 1
        else:
            lr_d_data_drv.set(0, 0)
            lr_d_H_drv.set(0, 0)
            lr_d_valid_drv.set(0, 0)
            
        if use_col2 and col2_idx < len(BATCH_4x2_DATA_COL2):
            val = BATCH_4x2_DATA_COL2[col2_idx]
            lr_d_data_drv.set(1, to_fixed(val))
            lr_d_H_drv.set(1, to_fixed(val)) # Drive H with same value
            lr_d_valid_drv.set(1, 1)
            col2_idx += 1
        else:
            lr_d_data_drv.set(1, 0)
            lr_d_H_drv.set(1, 0)
            lr_d_valid_drv.set(1, 0)
            
        await RisingEdge(dut.clk)

    lr_d_valid_drv.set(0, 0)
    lr_d_valid_drv.set(1, 0)
    
    for _ in range(5):
        await RisingEdge(dut.clk)
        
    monitor.kill()
    
    expected_col1 = [compute_leaky_relu_derivative(data, LEAK_FACTOR) for data in BATCH_4x2_DATA_COL1]
    for idx, (got, exp) in enumerate(zip(col1_results, expected_col1)):
        abs_err = abs(got - exp)
        assert abs_err <= 0.05, f"col1[{idx}]: error {abs_err:.5f} > 0.05"
    
    expected_col2 = [compute_leaky_relu_derivative(data, LEAK_FACTOR) for data in BATCH_4x2_DATA_COL2]
    for idx, (got, exp) in enumerate(zip(col2_results, expected_col2)):
        abs_err = abs(got - exp)
        assert abs_err <= 0.05, f"col2[{idx}]: error {abs_err:.5f} > 0.05"

@cocotb.test()
async def test_leaky_relu_derivative_parent_invalid_inputs(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    lr_d_valid_drv = PackedArrayDriver(dut.lr_d_valid_in, 1)
    lr_d_data_drv = PackedArrayDriver(dut.lr_d_data_in, 16)
    
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    
    dut.lr_leak_factor_in.value = to_fixed(LEAK_FACTOR)
    lr_d_data_drv.set(0, to_fixed(1.5))
    lr_d_data_drv.set(1, to_fixed(-2.0))
    # H doesn't matter for validity check
    
    lr_d_valid_drv.set(0, 0)
    lr_d_valid_drv.set(1, 0)
    await RisingEdge(dut.clk)
    await Timer(1, "ps")
    assert read_packed_valid(dut.lr_d_valid_out, 0) == 0
    
    lr_d_valid_drv.set(0, 1)
    lr_d_valid_drv.set(1, 0)
    await RisingEdge(dut.clk)
    await Timer(1, "ps")
    assert read_packed_valid(dut.lr_d_valid_out, 0) == 1
    assert read_packed_valid(dut.lr_d_valid_out, 1) == 0