#================DO NOT MODIFY BELOW===================== Compiler and simulator settings
IVERILOG = iverilog
VVP = vvp
# Use python to get the site-packages directory which contains cocotb
COCOTB_PREFIX = $(shell python3 -c "import cocotb; import os; print(os.path.dirname(os.path.dirname(cocotb.__file__)))")

COCOTB_LIBS = $(COCOTB_PREFIX)/cocotb/libs

SIM_BUILD_DIR = sim_build
SIM_VVP = $(SIM_BUILD_DIR)/sim.vvp

# Environment variables
export COCOTB_REDUCED_LOG_FMT=1
# Use absolute path for cocotb-config
export LIBPYTHON_LOC=$(shell /home/firatkizilboga/.local/bin/cocotb-config --libpython)
export PYTHONPATH := test:$(PYTHONPATH)
export PYGPI_PYTHON_BIN=$(shell which python3)

#=============== MODIFY BELOW ======================
# ********** IF YOU HAVE A NEW VERILOG FILE, ADD IT TO THE SOURCES VARIABLE
SOURCES = src/pe.sv \
          src/leaky_relu_child.sv \
          src/leaky_relu_derivative_child.sv \
          src/systolic.sv \
          src/bias_child.sv \
          src/fixedpoint.sv \
          src/control_unit.sv \
          src/unified_buffer.sv \
          src/vpu.sv \
		  src/loss_child.sv \
		  src/tpu.sv \
		  src/gradient_descent.sv \
		  src/input_skew_buffer.sv \
		  src/output_deskew_buffer.sv

# MODIFY 1) variable next to -s 
# MODIFY 2) variable next to $(SOURCES)
# MODIFY 3) variable right of COCOTB_TEST_MODULES=
# MODIFY 4) file name next to mv (i.e. pe.vcd)


# Test targets
test_pe: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s pe -s dump -g2012 $(SOURCES) test/dump_pe.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_pe $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv pe.vcd waveforms/ 2>/dev/null || true

test_systolic: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s systolic -s dump -g2012 $(SOURCES) test/dump_systolic.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_systolic $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv systolic.vcd waveforms/ 2>/dev/null || true

test_unified_buffer: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s unified_buffer -s dump -g2012 $(SOURCES) test/dump_unified_buffer.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_unified_buffer $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv unified_buffer.vcd waveforms/ 2>/dev/null || true

# Vector Processing unit test
test_vpu: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s vpu -s dump -g2012 $(SOURCES) test/dump_vpu.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_vpu $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv vpu.vcd waveforms/ 2>/dev/null || true

test_tpu: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s tpu -s dump -g2012 $(SOURCES) test/dump_tpu.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_tpu $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv tpu.vcd waveforms/ 2>/dev/null || true

test_gradient_descent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s gradient_descent -s dump -g2012 $(SOURCES) test/dump_gradient_descent.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_gradient_descent $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv gradient_descent.vcd waveforms/ 2>/dev/null || true

test_int8_e2e: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s tpu -s dump -g2012 $(SOURCES) test/dump_tpu.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_int8_e2e $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv tpu.vcd waveforms/int8_e2e.vcd 2>/dev/null || true

test_int4_e2e: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s tpu -s dump -g2012 $(SOURCES) test/dump_tpu.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_int4_e2e $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv tpu.vcd waveforms/int4_e2e.vcd 2>/dev/null || true

test_multi_precision: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s tpu -s dump -g2012 $(SOURCES) test/dump_tpu.sv
	PYTHONOPTIMIZE=$(NOASSERT) COCOTB_TEST_MODULES=test_multi_precision $(VVP) -M $(COCOTB_LIBS) -m libcocotbvpi_icarus $(SIM_VVP)
	! grep failure results.xml
	mv tpu.vcd waveforms/multi_precision.vcd 2>/dev/null || true

test_all: test_pe test_systolic test_unified_buffer test_vpu test_tpu test_gradient_descent test_multi_precision test_int8_e2e test_int4_e2e


# ============ DO NOT MODIFY BELOW THIS LINE ============== 

# Create simulation build directory and waveforms directory
$(SIM_BUILD_DIR):
	mkdir -p $(SIM_BUILD_DIR)
	mkdir -p waveforms

# Waveform viewing
show_%: waveforms/%.vcd waveforms/%.gtkw
	gtkwave $^ 

# Linting
lint:
	verible-verilog-lint src/*sv --rules_config verible.rules

# Cleanup
clean:
	rm -rf waveforms/*vcd $(SIM_BUILD_DIR) test/__pycache__

.PHONY: clean test_all test_multi_precision