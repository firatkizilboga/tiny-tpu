`timescale 1ns/1ps
`default_nettype none

module tpu #(
`ifdef TPU_ARRAY_SIZE
    parameter int SYSTOLIC_ARRAY_WIDTH = `TPU_ARRAY_SIZE
`else
    parameter int SYSTOLIC_ARRAY_WIDTH = 2
`endif
)(
    input logic clk,
    input logic rst,

    // UB wires (writing from host to UB)
    input logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_wr_host_data_in,
    input logic [SYSTOLIC_ARRAY_WIDTH-1:0] ub_wr_host_valid_in,

    // UB wires (inputting reading instructions from host)
    input logic ub_rd_start_in,
    input logic ub_rd_transpose, // Ignored in new Wide Memory design (Software transpose expected)
    input logic [8:0] ub_ptr_select,
    input logic [15:0] ub_rd_addr_in,
    input logic [15:0] ub_rd_row_size, // Ignored or mapped to count
    input logic [15:0] ub_rd_col_size, // Mapped to count

    // Learning rate
    input logic [15:0] learning_rate_in,

    // VPU data pathway
    input logic [3:0] vpu_data_pathway,

    input logic sys_switch_in,
    input logic [15:0] vpu_leak_factor_in,
    input logic [15:0] inv_batch_size_times_two_in,
    input logic [1:0] sys_mode
);
    
    // Unified Buffer Signals (Flat/Wide)
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_wr_data_in;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] ub_wr_valid_in;

    logic [15:0] ub_rd_col_size_out; // Not used in new UB? 
    // Wait, the new UB removed `ub_rd_col_size_out`. 
    // But `systolic.sv` needs `ub_rd_col_size_in` to enable PEs.
    // The Host instruction provides `ub_rd_col_size`. We should pass that through.
    // Or we rely on `ub_rd_count`?
    // `ub_rd_col_size` in instruction is usually "How many columns in matrix".
    // I will wire `ub_rd_col_size` directly to systolic?
    // But it needs to be valid only when reading inputs.
    // I will add a register to hold it or pass it through.
    
    // Actually, `ub_rd_col_size` logic in old UB was: 
    // `if (ptr_select==1 [Weights]) ub_rd_col_size_out = ...`
    // It sent col size when loading weights? 
    // Let's assume for now we pass the input `ub_rd_col_size` to the systolic array
    // latched when `ub_rd_start_in` is high and `ptr_select==1`.
    
    logic [15:0] latched_col_size;
    logic latched_col_size_valid;
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            latched_col_size <= 0;
            latched_col_size_valid <= 0;
        end else begin
            if (ub_rd_start_in && ub_ptr_select == 1) begin
                 latched_col_size <= ub_rd_col_size; // Or row_size depending on transpose?
                 // Since we removed transpose logic, assume direct mapping.
                 latched_col_size_valid <= 1;
            end else begin
                 latched_col_size_valid <= 0;
            end
        end
    end


    // UB Read Outputs (Flat)
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_rd_input_data;
    logic ub_rd_input_valid;
    
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_rd_weight_data;
    logic ub_rd_weight_valid;
    
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_rd_bias_data;
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_rd_Y_data;
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] ub_rd_H_data;

    // Skew Buffer Outputs (Skewed)
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] sys_input_data_skewed;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] sys_input_valid_skewed;
    
    // Switch Skew
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] sys_switch_flat;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] sys_switch_dummy_data; // Unused
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] sys_switch_skewed;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] sys_switch_valid_skewed; // Used as the switch signal?
    // Wait, input_skew_buffer skews DATA and VALID.
    // If we put {N{switch}} into DATA input of a skew buffer...
    // The DATA output will be skewed switch.
    assign sys_switch_flat = {SYSTOLIC_ARRAY_WIDTH{sys_switch_in}};

    // Systolic Outputs (Skewed)
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][31:0] sys_data_out;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] sys_valid_out;

    // VPU Outputs (Skewed)
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] vpu_data_out;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] vpu_valid_out;

    // Deskew Outputs (Flat)
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] deskewed_data_out;
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] deskewed_valid_out;

    assign ub_wr_data_in = deskewed_data_out;
    assign ub_wr_valid_in = deskewed_valid_out;

    // Unified Buffer
    unified_buffer #(
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) ub_inst (
        .clk(clk),
        .rst(rst),
        .ub_wr_data_in(ub_wr_data_in),
        .ub_wr_valid_in(ub_wr_valid_in),
        .ub_wr_host_data_in(ub_wr_host_data_in),
        .ub_wr_host_valid_in(ub_wr_host_valid_in),
        .ub_rd_start_in(ub_rd_start_in),
        .ub_ptr_select(ub_ptr_select),
        .ub_rd_addr_in(ub_rd_addr_in),
        .ub_rd_count(ub_rd_col_size), // Using col_size as count/length for now
        .learning_rate_in(learning_rate_in),
        .ub_rd_input_data_out(ub_rd_input_data),
        .ub_rd_input_valid_out(ub_rd_input_valid),
        .ub_rd_weight_data_out(ub_rd_weight_data),
        .ub_rd_weight_valid_out(ub_rd_weight_valid),
        .ub_rd_bias_data_out(ub_rd_bias_data),
        .ub_rd_Y_data_out(ub_rd_Y_data),
        .ub_rd_H_data_out(ub_rd_H_data)
    );

    // Input Skew Buffer (Data/Valid)
    input_skew_buffer #(
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) input_skew (
        .clk(clk),
        .rst(rst),
        .flat_data_in(ub_rd_input_data),
        .flat_valid_in(ub_rd_input_valid),
        .skewed_data_out(sys_input_data_skewed),
        .skewed_valid_out(sys_input_valid_skewed)
    );

    // Switch Skew Buffer
    // We treat 'switch' signal as 'data' to be skewed. 
    // 'valid' input is 1 (always valid).
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] switch_as_data_in;
    logic signed [SYSTOLIC_ARRAY_WIDTH-1:0][15:0] switch_as_data_out;
    
    genvar i;
    generate
        for (i=0; i<SYSTOLIC_ARRAY_WIDTH; i++) begin
            assign switch_as_data_in[i] = {15'b0, sys_switch_in};
            assign sys_switch_skewed[i] = switch_as_data_out[i][0];
        end
    endgenerate

    input_skew_buffer #(
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) switch_skew (
        .clk(clk),
        .rst(rst),
        .flat_data_in(switch_as_data_in),
        .flat_valid_in(1'b1),
        .skewed_data_out(switch_as_data_out),
        .skewed_valid_out() // unused
    );

    // Unpack weights for Systolic Array
    logic [15:0] sys_weight_in_unpacked [0:SYSTOLIC_ARRAY_WIDTH-1];
    logic sys_accept_w_unpacked [0:SYSTOLIC_ARRAY_WIDTH-1];
    
    generate
        for (i=0; i<SYSTOLIC_ARRAY_WIDTH; i++) begin
            assign sys_weight_in_unpacked[i] = ub_rd_weight_data[i];
            assign sys_accept_w_unpacked[i] = ub_rd_weight_valid; // Weights are loaded flat/simultaneously? 
            // If weights are stationary, we might not need to skew them.
            // UB outputs them flat.
            // If we load weights column by column...
            // The instruction reads weights.
            // Old UB had separate logic for weights.
        end
    endgenerate

    systolic #(
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) systolic_inst (
        .clk(clk),
        .rst(rst),
        .sys_data_in(sys_input_data_skewed),
        .sys_valid_in(sys_input_valid_skewed),
        .sys_switch_in(sys_switch_skewed),
        .sys_data_out(sys_data_out),
        .sys_valid_out(sys_valid_out),
        .sys_weight_in(sys_weight_in_unpacked),
        .sys_accept_w(sys_accept_w_unpacked),
        .ub_rd_col_size_in(latched_col_size),
        .ub_rd_col_size_valid_in(latched_col_size_valid),
        .sys_mode(sys_mode)
    );

    vpu #(
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) vpu_inst (
        .clk(clk),
        .rst(rst),
        .sys_mode(sys_mode),
        .requant_scale(16'b0),
        .requant_shift(16'b0),
        .requant_zero_point(16'b0),
        .vpu_data_pathway(vpu_data_pathway),
        .vpu_data_in(sys_data_out),
        .vpu_valid_in(sys_valid_out),
        .bias_scalar_in(ub_rd_bias_data),
        .lr_leak_factor_in(vpu_leak_factor_in),
        .Y_in(ub_rd_Y_data),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in),
        .H_in(ub_rd_H_data),
        .vpu_data_out(vpu_data_out),
        .vpu_valid_out(vpu_valid_out)
    );

    // Output Deskew Buffer
    output_deskew_buffer #(
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) output_deskew (
        .clk(clk),
        .rst(rst),
        .skewed_data_in(vpu_data_out),
        .skewed_valid_in(vpu_valid_out),
        .flat_data_out(deskewed_data_out),
        .flat_valid_out(deskewed_valid_out)
    );

endmodule
