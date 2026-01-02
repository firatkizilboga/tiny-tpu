`timescale 1ns/1ps
`default_nettype none

module output_deskew_buffer #(
    parameter int N = 2
)(
    input logic clk,
    input logic rst,
    input logic signed [N-1:0][15:0] skewed_data_in,
    input logic [N-1:0] skewed_valid_in,
    
    output logic signed [N-1:0][15:0] flat_data_out,
    output logic [N-1:0] flat_valid_out
);
    // Align to the LAST row (Row N-1).
    // Row N-1: 0 delay
    // Row i: (N-1) - i delay
    
    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_deskew
            localparam int DELAY = (N - 1) - i;
            
            if (DELAY == 0) begin
                assign flat_data_out[i] = skewed_data_in[i];
                assign flat_valid_out[i] = skewed_valid_in[i];
            end else begin
                logic signed [15:0] shift_reg_data [0:DELAY-1];
                logic shift_reg_valid [0:DELAY-1];
                
                always_ff @(posedge clk or posedge rst) begin
                    if (rst) begin
                        for (int k=0; k<DELAY; k++) begin
                            shift_reg_data[k] <= '0;
                            shift_reg_valid[k] <= 0;
                        end
                    end else begin
                        shift_reg_data[0] <= skewed_data_in[i];
                        shift_reg_valid[0] <= skewed_valid_in[i];
                        for (int k=1; k<DELAY; k++) begin
                            shift_reg_data[k] <= shift_reg_data[k-1];
                            shift_reg_valid[k] <= shift_reg_valid[k-1];
                        end
                    end
                end
                assign flat_data_out[i] = shift_reg_data[DELAY-1];
                assign flat_valid_out[i] = shift_reg_valid[DELAY-1];
            end
        end
    endgenerate
endmodule
