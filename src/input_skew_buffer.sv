`timescale 1ns/1ps
`default_nettype none

module input_skew_buffer #(
    parameter int N = 2
)(
    input logic clk,
    input logic rst,
    input logic signed [N-1:0][15:0] flat_data_in,
    input logic flat_valid_in, // Single valid for the whole vector
    
    output logic signed [N-1:0][15:0] skewed_data_out,
    output logic [N-1:0] skewed_valid_out // Independent valids per row
);
    // Row 0: No delay (pass through)
    // Row 1: 1 cycle delay
    // Row i: i cycles delay
    
    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_skew
            if (i == 0) begin
                assign skewed_data_out[0] = flat_data_in[0];
                assign skewed_valid_out[0] = flat_valid_in;
            end else begin
                logic signed [15:0] shift_reg_data [0:i-1];
                logic shift_reg_valid [0:i-1];
                
                always_ff @(posedge clk or posedge rst) begin
                    if (rst) begin
                        for (int k=0; k<i; k++) begin
                             shift_reg_data[k] <= '0;
                             shift_reg_valid[k] <= 0;
                        end
                    end else begin
                        shift_reg_data[0] <= flat_data_in[i];
                        shift_reg_valid[0] <= flat_valid_in;
                        for (int k=1; k<i; k++) begin
                             shift_reg_data[k] <= shift_reg_data[k-1];
                             shift_reg_valid[k] <= shift_reg_valid[k-1];
                        end
                    end
                end
                assign skewed_data_out[i] = shift_reg_data[i-1];
                assign skewed_valid_out[i] = shift_reg_valid[i-1];
            end
        end
    endgenerate
endmodule
