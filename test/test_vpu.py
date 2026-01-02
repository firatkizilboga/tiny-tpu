# Test these three pathways:
# (forward pass: hidden layer computations) input from sys --> bias --> leaky relu --> output
# (transition) input from sys --> bias --> leaky relu --> loss --> leaky relu derivative (handles a staggered hadamard product)--> output
# (backward pass) input from sys --> leaky relu derivative --> output

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

FRAC_BITS = 8
N_LANES = 2

def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF

def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)

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
            # Ensure val is masked to width
            val = self.values[i] & ((1 << self.width) - 1)
            full_val |= (val << (i * self.width))
        self.signal.value = full_val

Z1_pre = [
    [ 0, 0,],
    [-0.5792, 0.4234],
    [ 0.2985, 0.0913],
    [-0.2807, 0.5147],
]

B1 = [-0.4939,  0.189 ]

# H1 should be:
# [[-0.247   0.189 ]
#  [-0.5366  0.6124]
#  [-0.0977  0.2803]
#  [-0.3873  0.7037]]

Z2_pre = [
    [-0.0741],
    [-0.1014],
    [ 0.0315],
    [ 0.0042],
]

B2 = [0.6358]

Y = [
    [0.],
    [1.],
    [1.],
    [0.],
]


# H2 should be:
# [[0.5617]
#  [0.5344]
#  [0.6673]
#  [0.64  ]]

# i think this array is calculated from the cached H2 within the VPU. so i think we can delete it here in this testbench? - Evan
# dL_by_dH2 should look like 
# = [
#  [ 0.2808],
#  [-0.2328],
#  [-0.1664],
#  [ 0.32  ]
#  ]

dL_by_H1 = [
 [0.1479,  0.0831],
 [-0.1226, -0.0689],
 [-0.0876, -0.0492],
 [ 0.1685,  0.0947],
 ]

# so this means we need an h1 now i think to pair with the array above during the backwards pass test. 
# we need this array during backwwarsd pass cus it turnsint dH/dZ. then hadamarding it with the above will give us dL/dZ
H1 = [
 [-0.247,   0.189 ],
 [-0.5366,  0.6124],
 [-0.0977,  0.2803],
 [-0.3873,  0.7037]
 ]


@cocotb.test()
async def test_vector_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await RisingEdge(dut.clk)

    # Initialize drivers
    # vpu_data_in: 32-bit
    data_in_drv = PackedArrayDriver(dut.vpu_data_in, 32)
    # vpu_valid_in: 1-bit
    valid_in_drv = PackedArrayDriver(dut.vpu_valid_in, 1)
    # bias_scalar_in: 16-bit
    bias_drv = PackedArrayDriver(dut.bias_scalar_in, 16)
    # Y_in: 16-bit
    Y_drv = PackedArrayDriver(dut.Y_in, 16)
    # H_in: 16-bit
    H_drv = PackedArrayDriver(dut.H_in, 16)

    # Init signals
    data_in_drv.commit()
    valid_in_drv.commit()
    bias_drv.commit()
    Y_drv.commit()
    H_drv.commit()

    # Reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)

    ### Test forward pass pathway
    # ub/input -> bias -> lr -> ub/output
    dut.rst.value = 0
    dut.vpu_data_pathway.value = 0b1100 
    
    valid_in_drv.set(0, 1)
    valid_in_drv.set(1, 1)

    # Comes from UB
    bias_drv.set(0, to_fixed(B1[0]))
    bias_drv.set(1, to_fixed(B1[1]))
    dut.lr_leak_factor_in.value = to_fixed(0.5)
    
    for z in Z1_pre: # Load in z rows one at a time
        data_in_drv.set(0, to_fixed(z[0]))
        data_in_drv.set(1, to_fixed(z[1]))
        await RisingEdge(dut.clk)
    
    data_in_drv.set(0, to_fixed(0)) # reset everything
    data_in_drv.set(1, to_fixed(0))
    valid_in_drv.set(0, 0)
    valid_in_drv.set(1, 0)
    await ClockCycles(dut.clk, 10)


    ### Test transition pathway
    # ub/input -> bias -> lr -> loss -> lr_d -> ub/output
    dut.rst.value = 1
    await RisingEdge(dut.clk)

    dut.rst.value = 0
    dut.vpu_data_pathway.value = 0b1111
    valid_in_drv.set(0, 1)
    valid_in_drv.set(1, 1)

    bias_drv.set(0, to_fixed(B2[0]))
    # bias_drv.set(1, to_fixed(B2[1]))
    dut.lr_leak_factor_in.value = to_fixed(0.5)
    dut.inv_batch_size_times_two_in.value = to_fixed(2/4)  # 2/N where N is our batch size which is 4
    
    data_in_drv.set(0, to_fixed(Z2_pre[0][0]))
    # data_in_drv.set(1, to_fixed(z[1]))
    await RisingEdge(dut.clk)

    data_in_drv.set(0, to_fixed(Z2_pre[1][0]))
    # data_in_drv.set(1, to_fixed(z[1]))
    await RisingEdge(dut.clk)

    ## START PUTTING TARGET VALUES HERE??? IDK? if not, then shift it down by 1 clk cycle
    data_in_drv.set(0, to_fixed(Z2_pre[2][0]))
    # data_in_drv.set(1, to_fixed(z[1]))
    Y_drv.set(0, to_fixed(Y[0][0]))
    # Y_drv.set(1, to_fixed(Y[0][0]))
    await RisingEdge(dut.clk)


    data_in_drv.set(0, to_fixed(Z2_pre[3][0]))
    # data_in_drv.set(1, to_fixed(z[1]))
    Y_drv.set(0, to_fixed(Y[1][0]))
    # Y_drv.set(1, to_fixed(Y[0][0]))
    await RisingEdge(dut.clk)

    Y_drv.set(0, to_fixed(Y[2][0]))
    # Y_drv.set(1, to_fixed(Y[0][0]))
    valid_in_drv.set(0, 0)
    await RisingEdge(dut.clk)

    Y_drv.set(0, to_fixed(Y[3][0]))
    # Y_drv.set(1, to_fixed(Y[0][0]))
    await RisingEdge(dut.clk)
    
    valid_in_drv.set(0, 0)
    # valid_in_drv.set(1, 0)
    await ClockCycles(dut.clk, 10)




    # Test backward pass pathway
    # input from sys --> leaky relu derivative --> output (backward pass)
    dut.rst.value = 1
    await RisingEdge(dut.clk)

    # Test forward pass pathway
    dut.rst.value = 0
    dut.vpu_data_pathway.value = 0b0001 

    ## PREMATURELY start inputting Hs 1 clk cycle before we input the dL/dH values because it takes 1 clk cycle to compute its dH/dZ.
    ## ^^ maybe its a good assumption to have in future that activation derivatives are NOT combinational, but take multiple clk cycles. 
    await RisingEdge(dut.clk)

    # WE NEED TO INPUT A dL/dH here. and in the same clk cycle, we also need to input an H value, which becomes dH/dZ. 
    data_in_drv.set(0, to_fixed(dL_by_H1[0][0]))
    H_drv.set(0, to_fixed(H1[0][0]))

    valid_in_drv.set(0, 1)
    valid_in_drv.set(1, 0)

    await RisingEdge(dut.clk)
    data_in_drv.set(0, to_fixed(dL_by_H1[1][0]))
    data_in_drv.set(1, to_fixed(dL_by_H1[0][1]))

    H_drv.set(0, to_fixed(H1[1][0]))
    valid_in_drv.set(0, 1)
    valid_in_drv.set(1, 1)
    await RisingEdge(dut.clk)

    data_in_drv.set(0, to_fixed(dL_by_H1[2][0]))
    data_in_drv.set(1, to_fixed(dL_by_H1[1][1]))

    H_drv.set(0, to_fixed(H1[2][0]))
    H_drv.set(1, to_fixed(H1[1][1]))

    valid_in_drv.set(0, 1)
    valid_in_drv.set(1, 1)
    await RisingEdge(dut.clk)

    data_in_drv.set(0, to_fixed(dL_by_H1[3][0]))
    data_in_drv.set(1, to_fixed(dL_by_H1[2][1]))

    H_drv.set(0, to_fixed(H1[3][0]))
    H_drv.set(1, to_fixed(H1[2][1]))

    valid_in_drv.set(0, 1)
    valid_in_drv.set(1, 1)
    await RisingEdge(dut.clk)

    data_in_drv.set(1, to_fixed(dL_by_H1[3][1]))

    H_drv.set(1, to_fixed(H1[3][1]))

    valid_in_drv.set(0, 0)
    valid_in_drv.set(1, 1)


    await ClockCycles(dut.clk, 10)
