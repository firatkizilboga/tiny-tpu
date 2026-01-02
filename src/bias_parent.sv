`timescale 1ns/1ps
`default_nettype none

module bias_parent #(
    parameter int N = 2
)(
    input logic clk,
    input logic rst,

    input logic signed [N-1:0][15:0] bias_scalar_in,    // bias scalars fetched from the unified buffer (rename it to bias_scalar_ub_in)

    output logic [N-1:0] bias_Z_valid_out,

    input wire signed [N-1:0][15:0] bias_sys_data_in,

    input wire [N-1:0] bias_sys_valid_in,

    output logic signed [N-1:0][15:0] bias_z_data_out

); 
    // Each bias module handles a feature column for a pre-activation matrix. 

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : bias_columns
            bias_child column (
                .clk(clk),
                .rst(rst),
                .bias_scalar_in(bias_scalar_in[i]),
                .bias_Z_valid_out(bias_Z_valid_out[i]),
                .bias_sys_data_in(bias_sys_data_in[i]),
                .bias_sys_valid_in(bias_sys_valid_in[i]),
                .bias_z_data_out(bias_z_data_out[i])
            );
        end
    endgenerate

endmodule