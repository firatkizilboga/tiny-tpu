import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Utilities for packing/unpacking
def pack_int8_to_uint16(val_high, val_low):
    """Packs two 8-bit integers into one 16-bit word."""
    # Mask to 8 bits
    vh = val_high & 0xFF
    vl = val_low & 0xFF
    return (vh << 8) | vl

def pack_int4_to_uint16(v3, v2, v1, v0):
    """Packs four 4-bit integers into one 16-bit word."""
    return ((v3 & 0xF) << 12) | ((v2 & 0xF) << 8) | ((v1 & 0xF) << 4) | (v0 & 0xF)

async def reset_dut(dut):
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_int16_mode(dut):
    """Test INT16 Mode (Standard Matrix Multiplication)"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    
    dut.sys_mode.value = 1 # INT16 Mode
    
    # -----------------------------------------------------------
    # Test Case 1: Simple 32-bit Accumulation Test
    # -----------------------------------------------------------
    # We want to verify that the accumulation path is indeed 32-bit.
    # We will use the PE directly or the systolic array interface?
    # The DUT is 'tpu', so we have to drive the top-level signals.
    # Driving the TPU to load weights and inputs is complex.
    # For this unit-test level verification, we might want to peek 
    # directly into the PE behavior if possible, or assume the 
    # default "flow through" works if we set the right control signals.
    
    # Let's try to drive the systolic array inputs directly via the 
    # UB interface wires if they are accessible, OR drive the 
    # "ub_rd_input_data_out" signals if we force them (hacky).
    
    # Better approach: Use the standard TPU interface.
    # 1. Load Weights (Host -> UB)
    # 2. Drive Inputs (Host -> UB)
    # 3. Start Systolic Array
    
    # However, to save time and focus on the ALU logic, let's just 
    # verify the PE logic assuming the data transport works (which 
    # was verified by other tests). The most critical part is the 
    # PE ALU.
    
    # For this test, we will actually inspect the internal signals of 
    # the first PE (pe11) to verify the ALU operation.
    
    # Force PE11 inputs for quick verification
    # PE11 is at dut.systolic_inst.pe11
    
    pe = dut.systolic_inst.gen_row[0].gen_col[0].pe_inst
    
    # Enable PE
    pe.pe_enabled.value = 1
    
    # Test 1: INT16 Multiplication
    # 300 * 300 = 90000 (Fits in 32-bit, overflows 16-bit)
    
    # Load Weight: 300
    pe.pe_accept_w_in.value = 1
    pe.pe_weight_in.value = 300
    await RisingEdge(dut.clk)
    pe.pe_accept_w_in.value = 0
    
    # Move weight to active
    pe.pe_switch_in.value = 1
    await RisingEdge(dut.clk)
    pe.pe_switch_in.value = 0
    
    # Drive Input: 300
    pe.pe_input_in.value = 300
    pe.pe_valid_in.value = 1
    pe.pe_psum_in.value = 0
    
    await RisingEdge(dut.clk)
    
    # Check Result
    # Result should be available on pe_psum_out in the next cycle (registered)
    await RisingEdge(dut.clk) 
    
    result = pe.pe_psum_out.value.signed_integer
    dut._log.info(f"INT16 Test: 300 * 300 = {result}")
    
    assert result == 90000, f"INT16 Failed: Expected 90000, got {result}"

@cocotb.test()
async def test_int8_packed_mode(dut):
    """Test INT8 Packed Mode (W8A8)"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    
    dut.sys_mode.value = 2 # INT8 Packed Mode
    
    pe = dut.systolic_inst.gen_row[0].gen_col[0].pe_inst
    pe.pe_enabled.value = 1
    
    # -----------------------------------------------------------
    # Test Case 2: INT8 Packed SWAR
    # -----------------------------------------------------------
    # We want to multiply two pairs of numbers:
    # A = [10, -5]  (High, Low)
    # W = [2,   4]  (High, Low)
    # Expected: (10 * 2) + (-5 * 4) = 20 - 20 = 0
    
    # Pack Weights
    w_packed = pack_int8_to_uint16(10, -5) # Swap for packing logic check? 
    # Wait, function is pack(high, low).
    # If the PE splits [15:8] as high and [7:0] as low:
    # We want W_hi=2, W_lo=4.
    w_packed = pack_int8_to_uint16(2, 4)
    
    pe.pe_accept_w_in.value = 1
    pe.pe_weight_in.value = w_packed
    await RisingEdge(dut.clk)
    pe.pe_accept_w_in.value = 0
    
    pe.pe_switch_in.value = 1
    await RisingEdge(dut.clk)
    pe.pe_switch_in.value = 0
    
    # Pack Inputs: A_hi=10, A_lo=-5
    a_packed = pack_int8_to_uint16(10, -5)
    
    pe.pe_input_in.value = a_packed
    pe.pe_valid_in.value = 1
    pe.pe_psum_in.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    result = pe.pe_psum_out.value.signed_integer
    dut._log.info(f"INT8 Test: (10*2) + (-5*4) = {result}")
    
    assert result == 0, f"INT8 Failed: Expected 0, got {result}"
    
    # Test Case 2b: Overflow check within 16-bit
    # A = [100, 100]
    # W = [100, 100]
    # Res = 10000 + 10000 = 20000 (Fits in 16-bit, but intermediate prods fit in 16-bit)
    
    w_packed = pack_int8_to_uint16(100, 100)
    pe.pe_accept_w_in.value = 1
    pe.pe_weight_in.value = w_packed
    await RisingEdge(dut.clk)
    pe.pe_accept_w_in.value = 0
    pe.pe_switch_in.value = 1
    await RisingEdge(dut.clk)
    pe.pe_switch_in.value = 0
    
    a_packed = pack_int8_to_uint16(100, 100)
    pe.pe_input_in.value = a_packed
    pe.pe_valid_in.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    result = pe.pe_psum_out.value.signed_integer
    dut._log.info(f"INT8 Test 2: (100*100) + (100*100) = {result}")
    assert result == 20000, f"INT8 Failed: Expected 20000, got {result}"

    # -----------------------------------------------------------
    # Test Case 3: Randomized Stress Test (Acumulation over 50 cycles)
    # -----------------------------------------------------------
    
    # Reset PE for new test
    pe.pe_psum_in.value = 0
    pe.pe_valid_in.value = 0
    await RisingEdge(dut.clk)
    
    accumulated_expected = 0
    last_result = 0
    
    # We will feed 50 random vectors
    import random
    
    for i in range(50):
        # Generate random inputs
        a_hi = random.randint(-128, 127)
        a_lo = random.randint(-128, 127)
        w_hi = random.randint(-128, 127)
        w_lo = random.randint(-128, 127)
        
        # Calculate expected (SWAR dot product)
        expected_dot = (a_hi * w_hi) + (a_lo * w_lo)
        accumulated_expected += expected_dot
        
        # Pack
        w_packed = pack_int8_to_uint16(w_hi, w_lo)
        a_packed = pack_int8_to_uint16(a_hi, a_lo)
        
        # Load Weight
        pe.pe_accept_w_in.value = 1
        pe.pe_weight_in.value = w_packed
        await RisingEdge(dut.clk)
        pe.pe_accept_w_in.value = 0
        pe.pe_switch_in.value = 1
        await RisingEdge(dut.clk)
        pe.pe_switch_in.value = 0
        
        # Load Input (accumulating)
        pe.pe_input_in.value = a_packed
        pe.pe_valid_in.value = 1
        pe.pe_psum_in.value = last_result
            
        await RisingEdge(dut.clk)
        # Wait for computation (registered output)
        await RisingEdge(dut.clk) 
        
        last_result = int(pe.pe_psum_out.value.signed_integer)
        
    dut._log.info(f"INT8 Random Stress Test (50 cycles): Expected {accumulated_expected}, Got {last_result}")
    assert last_result == accumulated_expected, f"INT8 Stress Failed: Exp {accumulated_expected}, Got {last_result}"

@cocotb.test()
async def test_int4_packed_mode(dut):
    """Test INT4 Packed Mode (W4A4)"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    
    dut.sys_mode.value = 3 # INT4 Packed Mode
    
    pe = dut.systolic_inst.gen_row[0].gen_col[0].pe_inst
    pe.pe_enabled.value = 1
    
    # -----------------------------------------------------------
    # Test Case 3: INT4 Packed Sign Extension
    # -----------------------------------------------------------
    # We want to verify that 0xF is treated as -1, not 15.
    # A = [-1, -1, -1, -1] = 0xFFFF
    # W = [ 1,  1,  1,  1] = 0x1111
    # Res = (-1*1) * 4 = -4
    
    w_packed = pack_int4_to_uint16(1, 1, 1, 1) # 0x1111
    pe.pe_accept_w_in.value = 1
    pe.pe_weight_in.value = w_packed
    await RisingEdge(dut.clk)
    pe.pe_accept_w_in.value = 0
    pe.pe_switch_in.value = 1
    await RisingEdge(dut.clk)
    pe.pe_switch_in.value = 0
    
    a_packed = pack_int4_to_uint16(-1, -1, -1, -1) # 0xFFFF
    pe.pe_input_in.value = a_packed
    pe.pe_valid_in.value = 1
    pe.pe_psum_in.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    result = pe.pe_psum_out.value.signed_integer
    dut._log.info(f"INT4 Test 1: (-1*1)*4 = {result}")
    
    assert result == -4, f"INT4 Failed: Expected -4, got {result}. This likely means sign extension failed!"
    
    w_packed = pack_int4_to_uint16(1, -1, 5, 2)
    pe.pe_accept_w_in.value = 1
    pe.pe_weight_in.value = w_packed
    await RisingEdge(dut.clk)
    pe.pe_accept_w_in.value = 0
    pe.pe_switch_in.value = 1
    await RisingEdge(dut.clk)
    pe.pe_switch_in.value = 0
    
    a_packed = pack_int4_to_uint16(-8, 7, 0, -1)
    pe.pe_input_in.value = a_packed
    pe.pe_valid_in.value = 1
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    result = pe.pe_psum_out.value.signed_integer
    dut._log.info(f"INT4 Test 2: Complex = {result}")
    
    assert result == -17, f"INT4 Failed: Expected -17, got {result}"

    # -----------------------------------------------------------
    # Test Case 3c: Randomized Stress Test (INT4)
    # -----------------------------------------------------------
    import random
    accumulated_expected = 0
    
    # We loop result back to input manually to simulate accumulation chain
    # Note: pe_psum_out is registered delay 1 cycle from ALU.
    # In this test loop, we wait 2 cycles: one for loading psum_in, one for output validity.
    # Actually wait, in each loop iteration:
    # Cycle 0: drive inputs (W, A, Psum_in)
    # Cycle 1: PE registers inputs
    # Cycle 2: PE registers output (pe_psum_out valid)
    
    # We need to grab result from previous iteration to feed into next.
    last_result = 0
    
    for i in range(50):
        # random 4-bit integers (-8 to 7)
        a_s = [random.randint(-8, 7) for _ in range(4)]
        w_s = [random.randint(-8, 7) for _ in range(4)]
        
        # Expected
        dot = sum([a*w for a, w in zip(a_s, w_s)])
        accumulated_expected += dot
        
        # Pack
        w_packed = pack_int4_to_uint16(w_s[3], w_s[2], w_s[1], w_s[0])
        a_packed = pack_int4_to_uint16(a_s[3], a_s[2], a_s[1], a_s[0])
        
        # Load Weight
        pe.pe_accept_w_in.value = 1
        pe.pe_weight_in.value = w_packed
        await RisingEdge(dut.clk)
        pe.pe_accept_w_in.value = 0
        pe.pe_switch_in.value = 1
        await RisingEdge(dut.clk)
        pe.pe_switch_in.value = 0
        
        # Load Input
        pe.pe_input_in.value = a_packed
        pe.pe_valid_in.value = 1
        pe.pe_psum_in.value = last_result
        
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        
        last_result = int(pe.pe_psum_out.value.signed_integer)
        
    dut._log.info(f"INT4 Random Stress Test (50 cycles): Expected {accumulated_expected}, Got {last_result}")
    assert last_result == accumulated_expected, f"INT4 Stress Failed: Exp {accumulated_expected}, Got {last_result}"
