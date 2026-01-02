import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

FRAC_BITS = 8
N_LANES = 2

def to_fixed(val, frac_bits=FRAC_BITS):
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF

def from_fixed(val, frac_bits=FRAC_BITS):
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)

def compute_gradient(h_val, y_val, batch_size):
    return 2.0 * (h_val - y_val) / batch_size

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
        await ReadOnly() # Wait for signals to settle
        
        if read_packed_valid(dut.valid_out, 0):
            val = read_packed(dut.gradient_out, 0, 16)
            col1_results.append(from_fixed(val))
            
        if read_packed_valid(dut.valid_out, 1):
            val = read_packed(dut.gradient_out, 1, 16)
            col2_results.append(from_fixed(val))

# test data
BATCH_4x2_H_COL1 = [0.7, 0.5, 0.3, 0.9]
BATCH_4x2_Y_COL1 = [1.0, 0.0, 0.5, 1.0]
BATCH_4x2_H_COL2 = [0.8, 0.6, 0.2, 0.4]
BATCH_4x2_Y_COL2 = [0.0, 1.0, 0.3, 0.7]

H_VALUES = [0.6831, 0.806, 0.4905, 0.5487]
Y_VALUES = [0.0, 1.0, 1.0, 0.0]

@cocotb.test()
async def test_loss_parent_4x2_staggered(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    H_drv = PackedArrayDriver(dut.H_in, 16)
    Y_drv = PackedArrayDriver(dut.Y_in, 16)
    valid_drv = PackedArrayDriver(dut.valid_in, 1)

    # reset
    dut.rst.value = 1
    H_drv.commit()
    Y_drv.commit()
    valid_drv.commit()
    dut.inv_batch_size_times_two_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    
    inv_n_times_2 = to_fixed(0.5)
    dut.inv_batch_size_times_two_in.value = inv_n_times_2
    
    col1_results = []
    col2_results = []
    
    # Start monitor
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
        if use_col1 and col1_idx < len(BATCH_4x2_H_COL1):
            H_drv.set(0, to_fixed(BATCH_4x2_H_COL1[col1_idx]))
            Y_drv.set(0, to_fixed(BATCH_4x2_Y_COL1[col1_idx]))
            valid_drv.set(0, 1)
            col1_idx += 1
        else:
            H_drv.set(0, 0)
            Y_drv.set(0, 0)
            valid_drv.set(0, 0)
            
        if use_col2 and col2_idx < len(BATCH_4x2_H_COL2):
            H_drv.set(1, to_fixed(BATCH_4x2_H_COL2[col2_idx]))
            Y_drv.set(1, to_fixed(BATCH_4x2_Y_COL2[col2_idx]))
            valid_drv.set(1, 1)
            col2_idx += 1
        else:
            H_drv.set(1, 0)
            Y_drv.set(1, 0)
            valid_drv.set(1, 0)
            
        await RisingEdge(dut.clk)

    # Clean inputs
    valid_drv.set(0, 0)
    valid_drv.set(1, 0)
    
    # Wait for pipeline flush
    for _ in range(5):
        await RisingEdge(dut.clk)
        
    monitor.kill()
    
    expected_col1 = [compute_gradient(h, y, 4) for h, y in zip(BATCH_4x2_H_COL1, BATCH_4x2_Y_COL1)]
    for idx, (got, exp) in enumerate(zip(col1_results, expected_col1)):
        rel_err = abs(got - exp) / max(abs(exp), 1e-6)
        # Relax tolerance slightly due to fixed point quantization
        assert rel_err <= 0.15, f"col1[{idx}]: error {rel_err:.3f} > 15%"
    
    expected_col2 = [compute_gradient(h, y, 4) for h, y in zip(BATCH_4x2_H_COL2, BATCH_4x2_Y_COL2)]
    for idx, (got, exp) in enumerate(zip(col2_results, expected_col2)):
        rel_err = abs(got - exp) / max(abs(exp), 1e-6)
        assert rel_err <= 0.15, f"col2[{idx}]: error {rel_err:.3f} > 15%"

@cocotb.test()
async def test_loss_parent_as_single_child(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    H_drv = PackedArrayDriver(dut.H_in, 16)
    Y_drv = PackedArrayDriver(dut.Y_in, 16)
    valid_drv = PackedArrayDriver(dut.valid_in, 1)

    dut.rst.value = 1
    H_drv.commit()
    Y_drv.commit()
    valid_drv.commit()
    dut.inv_batch_size_times_two_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    
    inv_n_times_2 = to_fixed(0.5)
    dut.inv_batch_size_times_two_in.value = inv_n_times_2
    
    col1_results = [] # Reuse same list but ignore col2
    col2_results = [] # unused
    
    monitor = cocotb.start_soon(monitor_outputs(dut, col1_results, col2_results))
    
    for idx, (h_val, y_val) in enumerate(zip(H_VALUES, Y_VALUES)):
        H_drv.set(0, to_fixed(h_val))
        Y_drv.set(0, to_fixed(y_val))
        valid_drv.set(0, 1)
        valid_drv.set(1, 0)
        await RisingEdge(dut.clk)
    
    valid_drv.set(0, 0)
    
    for _ in range(5):
        await RisingEdge(dut.clk)
        
    monitor.kill()
    
    expected_gradients = [compute_gradient(h, y, 4) for h, y in zip(H_VALUES, Y_VALUES)]
    for idx, (got, exp) in enumerate(zip(col1_results, expected_gradients)):
        rel_err = abs(got - exp) / max(abs(exp), 1e-6)
        assert rel_err <= 0.15, f"result[{idx}]: expected {exp:.5f}, got {got:.5f}, error {rel_err:.3f} > 15%"