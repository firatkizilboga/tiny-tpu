"""
Utility classes for driving packed array signals in cocotb tests.
These are used to interface with N-parametric SystemVerilog modules.
"""

class PackedArrayDriver:
    """
    Driver for packed array signals like [N-1:0][WIDTH-1:0].
    Allows setting individual lanes and commits them as a single packed value.
    """
    def __init__(self, signal, width, n_lanes=2):
        self.signal = signal
        self.width = width
        self.n_lanes = n_lanes
        self.values = [0] * n_lanes

    def set(self, idx, val):
        """Set a single lane value and commit."""
        self.values[idx] = val
        self.commit()

    def set_all(self, vals):
        """Set all lane values and commit."""
        for i, v in enumerate(vals):
            if i < self.n_lanes:
                self.values[i] = v
        self.commit()

    def commit(self):
        """Commit all values to the signal as a packed value."""
        full_val = 0
        for i in range(self.n_lanes):
            val = int(self.values[i]) & ((1 << self.width) - 1)
            full_val |= (val << (i * self.width))
        self.signal.value = full_val


def read_packed_data(signal, idx, width=16, signed=True):
    """Read a single lane from a packed array signal."""
    try:
        full_val = signal.value.integer
    except ValueError:
        return 0
    mask = (1 << width) - 1
    val = (full_val >> (idx * width)) & mask
    if signed and val >= (1 << (width - 1)):
        val -= (1 << width)
    return val


def read_packed_valid(signal, idx):
    """Read a single valid bit from a packed valid signal."""
    try:
        full_val = signal.value.integer
    except ValueError:
        return 0
    return (full_val >> idx) & 1

def to_fixed(val, frac_bits=8):
    """Convert a float to 16-bit fixed point with 8 fractional bits."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF

def to_fixed_32(val, frac_bits=16):
    """convert python float to signed 32-bit fixed-point (for VPU input which is sys array output)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFFFFFF

def from_fixed(val, frac_bits=8):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)
