`timescale 1ns/1ps
`default_nettype none

// NxN Parametric Systolic Array
// 
// Topology (weight-stationary dataflow):
// - Inputs flow horizontally (left-to-right) across rows
// - PSums flow vertically (top-to-bottom) down columns  
// - Weights are loaded from top and propagate down columns
// - Valid signals and Switch signals propagate horizontally (left-to-right) along with Data.
//   (Requires inputs to be skewed/staggered externally)
//
module systolic #(
    parameter int N = 2
)(
    input logic clk,
    input logic rst,

    // Input signals from left side (Skewed)
    input logic signed [N-1:0][15:0] sys_data_in,
    input logic [N-1:0] sys_valid_in,
    input logic [N-1:0] sys_switch_in, 

    // Output signals from bottom
    output logic signed [N-1:0][31:0] sys_data_out,
    output logic [N-1:0] sys_valid_out,

    // Weight signals from top
    input logic [15:0] sys_weight_in [0:N-1], // Unpacked to match single-word width per col
    input logic sys_accept_w [0:N-1],

    input logic [15:0] ub_rd_col_size_in,
    input logic ub_rd_col_size_valid_in,

    input logic [1:0] sys_mode
);

    // Internal wires
    logic signed [15:0] pe_input_out [0:N-1][0:N-1];
    logic signed [31:0] pe_psum_out [0:N-1][0:N-1];
    logic [15:0] pe_weight_out [0:N-1][0:N-1];
    logic pe_switch_out [0:N-1][0:N-1];
    logic pe_valid_out [0:N-1][0:N-1];

    logic [N-1:0] pe_enabled;

    genvar row, col;
    generate
        for (row = 0; row < N; row++) begin : gen_row
            for (col = 0; col < N; col++) begin : gen_col
                
                logic [15:0] pe_input_in_wire;
                logic signed [31:0] pe_psum_in_wire;
                logic [15:0] pe_weight_in_wire;
                logic pe_valid_in_wire;
                logic pe_switch_in_wire;
                logic pe_accept_w_wire;

                // INPUT/VALID/SWITCH: From Left
                if (col == 0) begin : leftmost
                    assign pe_input_in_wire = sys_data_in[row];
                    assign pe_valid_in_wire = sys_valid_in[row];
                    assign pe_switch_in_wire = sys_switch_in[row];
                end else begin : not_leftmost
                    assign pe_input_in_wire = pe_input_out[row][col-1];
                    assign pe_valid_in_wire = pe_valid_out[row][col-1];
                    assign pe_switch_in_wire = pe_switch_out[row][col-1];
                end

                // PSUM: From Top
                if (row == 0) begin : topmost
                    assign pe_psum_in_wire = 32'b0;
                end else begin : not_topmost
                    assign pe_psum_in_wire = pe_psum_out[row-1][col];
                end

                // WEIGHT: From Top
                if (row == 0) begin : weight_top
                    assign pe_weight_in_wire = sys_weight_in[col];
                end else begin : weight_chain
                    assign pe_weight_in_wire = pe_weight_out[row-1][col];
                end
                
                assign pe_accept_w_wire = sys_accept_w[col];

                pe pe_inst (
                    .clk(clk),
                    .rst(rst),
                    .pe_enabled(pe_enabled[col]),

                    .pe_valid_in(pe_valid_in_wire),
                    .pe_valid_out(pe_valid_out[row][col]),

                    .pe_accept_w_in(pe_accept_w_wire),
                    .pe_switch_in(pe_switch_in_wire),
                    .pe_switch_out(pe_switch_out[row][col]),

                    .pe_input_in(pe_input_in_wire),
                    .pe_psum_in(pe_psum_in_wire),
                    .pe_weight_in(pe_weight_in_wire),
                    .pe_input_out(pe_input_out[row][col]),
                    .pe_psum_out(pe_psum_out[row][col]),
                    .pe_weight_out(pe_weight_out[row][col]),
                    .sys_mode(sys_mode)
                );

            end
        end
    endgenerate

    // Outputs
    generate
        for (genvar i = 0; i < N; i++) begin : gen_outputs
            assign sys_data_out[i] = pe_psum_out[N-1][i];
            assign sys_valid_out[i] = pe_valid_out[N-1][i];
        end
    endgenerate

    // PE Enable Logic
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            pe_enabled <= '0;
        end else begin
            if (ub_rd_col_size_valid_in) begin
                pe_enabled <= (1 << ub_rd_col_size_in) - 1;
            end
        end
    end

endmodule
