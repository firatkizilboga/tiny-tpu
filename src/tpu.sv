module tpu #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2
)(
    input logic clk,
    input logic rst,

    // UB wires (writing from host to UB)
    input logic [15:0] ub_wr_host_data_in [0:SYSTOLIC_ARRAY_WIDTH-1],
    input logic ub_wr_host_valid_in [0:SYSTOLIC_ARRAY_WIDTH-1],

    // UB wires (inputting reading instructions from host)
    input logic ub_rd_start_in,
    input logic ub_rd_transpose,
    input logic [8:0] ub_ptr_select,
    input logic [15:0] ub_rd_addr_in,
    input logic [15:0] ub_rd_row_size,
    input logic [15:0] ub_rd_col_size,

    // Learning rate
    input logic [15:0] learning_rate_in,

    // VPU data pathway
    input logic [3:0] vpu_data_pathway,

    input logic sys_switch_in,
    input logic [15:0] vpu_leak_factor_in,
    input logic [15:0] inv_batch_size_times_two_in,
    input logic [1:0] sys_mode
);
    // UB internal output wires
    logic [15:0] ub_wr_data_in [0:SYSTOLIC_ARRAY_WIDTH-1];
    logic ub_wr_valid_in [0:SYSTOLIC_ARRAY_WIDTH-1];

    // Number of columns in the matrix to send to systolic array to disable columns of PEs
    logic [15:0] ub_rd_col_size_out;
    logic ub_rd_col_size_valid_out;
    
    // Array wires for connection to unified_buffer (now using arrays)
    logic [15:0] ub_rd_input_data_out [0:SYSTOLIC_ARRAY_WIDTH-1];
    logic ub_rd_input_valid_out [0:SYSTOLIC_ARRAY_WIDTH-1];

    logic [15:0] ub_rd_weight_data_out [0:SYSTOLIC_ARRAY_WIDTH-1];
    logic ub_rd_weight_valid_out [0:SYSTOLIC_ARRAY_WIDTH-1];

    logic [15:0] ub_rd_bias_data_out [0:SYSTOLIC_ARRAY_WIDTH-1];
    logic [15:0] ub_rd_Y_data_out [0:SYSTOLIC_ARRAY_WIDTH-1];
    logic [15:0] ub_rd_H_data_out [0:SYSTOLIC_ARRAY_WIDTH-1];

    // Systolic array internal output wires (now arrays)
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][31:0] sys_data_out;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] sys_valid_out;

    // VPU internal output wires
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] vpu_data_out;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] vpu_valid_out;

    // Connect VPU outputs to UB write inputs
    // Assuming 1-to-1 mapping
    generate
        for (genvar i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin : gen_ub_wr_conn
            assign ub_wr_data_in[i] = vpu_data_out[i];
            assign ub_wr_valid_in[i] = vpu_valid_out[i];
        end
    endgenerate
    
    unified_buffer #(
        .SYSTOLIC_ARRAY_WIDTH(SYSTOLIC_ARRAY_WIDTH)
    ) ub_inst(
        .clk(clk),
        .rst(rst),

        .ub_wr_data_in(ub_wr_data_in),
        .ub_wr_valid_in(ub_wr_valid_in),

        // Write ports from host to UB (for loading in parameters)
        .ub_wr_host_data_in(ub_wr_host_data_in),
        .ub_wr_host_valid_in(ub_wr_host_valid_in),

        // Read instruction input from instruction memory
        .ub_rd_start_in(ub_rd_start_in),
        .ub_rd_transpose(ub_rd_transpose),
        .ub_ptr_select(ub_ptr_select),
        .ub_rd_addr_in(ub_rd_addr_in),
        .ub_rd_row_size(ub_rd_row_size),
        .ub_rd_col_size(ub_rd_col_size),

        // Learning rate input
        .learning_rate_in(learning_rate_in),

        // Read ports from UB to left side of systolic array
        .ub_rd_input_data_out_0(ub_rd_input_data_out[0]),
        .ub_rd_input_data_out_1(ub_rd_input_data_out[1]),
        .ub_rd_input_valid_out_0(ub_rd_input_valid_out[0]),
        .ub_rd_input_valid_out_1(ub_rd_input_valid_out[1]),

        // Read ports from UB to top of systolic array
        .ub_rd_weight_data_out_0(ub_rd_weight_data_out[0]),
        .ub_rd_weight_data_out_1(ub_rd_weight_data_out[1]),
        .ub_rd_weight_valid_out_0(ub_rd_weight_valid_out[0]),
        .ub_rd_weight_valid_out_1(ub_rd_weight_valid_out[1]),

        // Read ports from UB to bias modules in VPU
        .ub_rd_bias_data_out_0(ub_rd_bias_data_out[0]),
        .ub_rd_bias_data_out_1(ub_rd_bias_data_out[1]),

        // Read ports from UB to loss modules (Y matrices) in VPU
        .ub_rd_Y_data_out_0(ub_rd_Y_data_out[0]),
        .ub_rd_Y_data_out_1(ub_rd_Y_data_out[1]),

        // Read ports from UB to activation derivative modules (H matrices) in VPU
        .ub_rd_H_data_out_0(ub_rd_H_data_out[0]),
        .ub_rd_H_data_out_1(ub_rd_H_data_out[1]),

        // Outputs to send number of columns to systolic array
        .ub_rd_col_size_out(ub_rd_col_size_out),
        .ub_rd_col_size_valid_out(ub_rd_col_size_valid_out)
    );

    systolic #(
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) systolic_inst (
        .clk(clk),
        .rst(rst),

        // Input signals from left side of systolic array (array interface)
        .sys_data_in(ub_rd_input_data_out),
        .sys_start(ub_rd_input_valid_out[0]),

        // Output signals from bottom of systolic array (array interface)
        .sys_data_out(sys_data_out),
        .sys_valid_out(sys_valid_out),

        // Weight signals from top of systolic array (array interface)
        .sys_weight_in(ub_rd_weight_data_out),
        .sys_accept_w(ub_rd_weight_valid_out),

        .sys_switch_in(sys_switch_in),

        .ub_rd_col_size_in(ub_rd_col_size_out),
        .ub_rd_col_size_valid_in(ub_rd_col_size_valid_out),
        .sys_mode(sys_mode)
    );

    // Cast the UB outputs to signed arrays for VPU compat
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_rd_bias_data_signed;
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_rd_Y_data_signed;
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_rd_H_data_signed;
    
    generate
        for (genvar i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin : gen_casts
            assign ub_rd_bias_data_signed[i] = $signed(ub_rd_bias_data_out[i]);
            assign ub_rd_Y_data_signed[i]    = $signed(ub_rd_Y_data_out[i]);
            assign ub_rd_H_data_signed[i]    = $signed(ub_rd_H_data_out[i]);
        end
    endgenerate

    vpu #(
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) vpu_inst (
        .clk(clk),
        .rst(rst),
        .sys_mode(sys_mode),
        .requant_scale(16'b0), // TODO: Wire these out
        .requant_shift(16'b0), // TODO: Wire these out
        .requant_zero_point(16'b0), // TODO: Wire these out

        .vpu_data_pathway(vpu_data_pathway), // 4-bits to signify which modules to route the inputs to (1 bit for each module)

        // Inputs from systolic array (using array indices)
        .vpu_data_in(sys_data_out),
        .vpu_valid_in(sys_valid_out),

        // Inputs from UB
        .bias_scalar_in(ub_rd_bias_data_signed),
        .lr_leak_factor_in(vpu_leak_factor_in),
        .Y_in(ub_rd_Y_data_signed),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in),
        .H_in(ub_rd_H_data_signed),

        // Outputs to UB
        .vpu_data_out(vpu_data_out),
        .vpu_valid_out(vpu_valid_out)
    ); 

    // DEBUG: Trace TPU level signals
    always @(posedge clk) begin
        if (sys_valid_out != 0) begin
             $display("[TPU] t=%0t: sys_valid_out!=0 (some bits set)", $time);
        end 

        if (vpu_valid_out != 0) begin
             $display("[TPU] t=%0t: vpu_valid_out!=0 (some bits set)", $time);
        end
    end

endmodule