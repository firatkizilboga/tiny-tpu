`timescale 1ns/1ps
`default_nettype none

// NxN Parametric Systolic Array
// 
// Topology (weight-stationary dataflow):
// - Inputs flow horizontally (left-to-right) across rows
// - PSums flow vertically (top-to-bottom) down columns  
// - Weights are loaded from top and propagate down columns
// - Valid signals propagate DIAGONALLY (from PE[r,c] to PE[r+1,c] and PE[r,c+1])
//
// For a 2x2 array:
//         col0       col1
//   row0: PE[0,0] -> PE[0,1]   (valid propagates right)
//            ↓          ↓      (psum flows down, valid also goes diagonally)
//   row1: PE[1,0] -> PE[1,1]   (outputs)
//
module systolic #(
    parameter int N = 2  // Array dimension (NxN PEs)
)(
    input logic clk,
    input logic rst,

    // Input signals from left side of systolic array (N inputs, one per row)
    input logic [15:0] sys_data_in [0:N-1],
    input logic sys_start,

    // Output signals from bottom of systolic array (N outputs, one per column)
    output logic signed [N-1:0][31:0] sys_data_out,
    output logic [N-1:0] sys_valid_out,

    // Weight signals from top of systolic array (N weights, one per column)
    input logic [15:0] sys_weight_in [0:N-1],
    input logic sys_accept_w [0:N-1],

    input logic sys_switch_in,

    input logic [15:0] ub_rd_col_size_in,
    input logic ub_rd_col_size_valid_in,

    input logic [1:0] sys_mode
);

    // Internal wires for PE interconnections
    // Horizontal: input_out flows left-to-right within each row
    logic [15:0] pe_input_out [0:N-1][0:N-1];
    
    // Vertical: psum_out flows top-to-bottom within each column
    logic signed [31:0] pe_psum_out [0:N-1][0:N-1];
    
    // Vertical: weight_out flows top-to-bottom within each column
    logic [15:0] pe_weight_out [0:N-1][0:N-1];
    
    // Switch signals propagate from top-left to bottom-right
    logic pe_switch_out [0:N-1][0:N-1];
    
    // Valid signals propagate DIAGONALLY (right and down from each PE)
    logic pe_valid_out [0:N-1][0:N-1];

    // PE enable mask (based on column size)
    logic [N-1:0] pe_enabled;

    // Generate NxN PE array
    genvar row, col;
    generate
        for (row = 0; row < N; row++) begin : gen_row
            for (col = 0; col < N; col++) begin : gen_col
                
                // Compute input connections based on position
                logic [15:0] pe_input_in_wire;
                logic signed [31:0] pe_psum_in_wire;
                logic [15:0] pe_weight_in_wire;
                logic pe_valid_in_wire;
                logic pe_switch_in_wire;
                logic pe_accept_w_wire;

                // INPUT: comes from left neighbor, or from sys_data_in if leftmost column
                if (col == 0) begin : leftmost
                    assign pe_input_in_wire = sys_data_in[row];
                end else begin : not_leftmost
                    assign pe_input_in_wire = pe_input_out[row][col-1];
                end

                // PSUM: comes from top neighbor, or 0 if topmost row
                if (row == 0) begin : topmost
                    assign pe_psum_in_wire = 32'b0;
                end else begin : not_topmost
                    assign pe_psum_in_wire = pe_psum_out[row-1][col];
                end

                // WEIGHT: comes from top neighbor, or from sys_weight_in if topmost row
                if (row == 0) begin : weight_top
                    assign pe_weight_in_wire = sys_weight_in[col];
                end else begin : weight_chain
                    assign pe_weight_in_wire = pe_weight_out[row-1][col];
                end
                
                // Accept weight signal is per-column
                assign pe_accept_w_wire = sys_accept_w[col];

                // VALID: propagates right on top row, then down each column
                // Original pattern:
                //   PE11's valid → PE12 (right) AND PE21 (down)
                //   PE12's valid → PE22 (down)
                // So: top row gets from left, all other rows get from above
                if (row == 0 && col == 0) begin : valid_corner
                    assign pe_valid_in_wire = sys_start;
                end else if (row == 0) begin : valid_top_edge
                    // Top row: valid propagates from left neighbor
                    assign pe_valid_in_wire = pe_valid_out[0][col-1];
                end else begin : valid_from_above
                    // All other rows: valid propagates from PE directly above (same column)
                    assign pe_valid_in_wire = pe_valid_out[row-1][col];
                end

                // SWITCH: propagates same pattern as valid (right on top row, down columns)
                if (row == 0 && col == 0) begin : switch_corner
                    assign pe_switch_in_wire = sys_switch_in;
                end else if (row == 0) begin : switch_top_edge
                    assign pe_switch_in_wire = pe_switch_out[0][col-1];
                end else begin : switch_from_above
                    // All other rows: switch propagates from PE directly above (same column)
                    assign pe_switch_in_wire = pe_switch_out[row-1][col];
                end

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

    // Connect bottom row outputs to module outputs
    generate
        for (genvar i = 0; i < N; i++) begin : gen_outputs
            assign sys_data_out[i] = pe_psum_out[N-1][i];
            assign sys_valid_out[i] = pe_valid_out[N-1][i];
        end
    endgenerate

    // PE enable logic based on column size
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            pe_enabled <= '0;
        end else begin
            if (ub_rd_col_size_valid_in) begin
                pe_enabled <= (1 << ub_rd_col_size_in) - 1;
            end
        end
    end

    // DEBUG 
    always @(posedge clk) begin
        if (ub_rd_col_size_valid_in)
            $display("[SYS] t=%0t: pe_enabled set to %b", $time, (1 << ub_rd_col_size_in) - 1);
        if (sys_start && pe_enabled != 0)
            $display("[SYS] t=%0t: sys_start, pe_enabled=%b, sys_mode=%d", $time, pe_enabled, sys_mode);
        if (sys_valid_out[0] || sys_valid_out[1])
            $display("[SYS] t=%0t: sys_valid_out=[%b,%b]", $time, sys_valid_out[0], sys_valid_out[1]);
    end

endmodule