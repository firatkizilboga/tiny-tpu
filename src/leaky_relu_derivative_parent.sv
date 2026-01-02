`timescale 1ns/1ps
`default_nettype none

module leaky_relu_derivative_parent #(
    parameter int N = 2
)(
    input logic clk,
    input logic rst,
    input logic [N-1:0] lr_d_valid_in,
    input logic signed [N-1:0][15:0] lr_d_data_in,
    input logic signed [N-1:0][15:0] lr_d_H_in,
    input logic signed [15:0] lr_leak_factor_in,

    output logic signed [N-1:0][15:0] lr_d_data_out,
    output logic [N-1:0] lr_d_valid_out
);

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : lr_d_cols
            leaky_relu_derivative_child leaky_relu_derivative_col (
                .clk(clk),
                .rst(rst),
                .lr_d_valid_in(lr_d_valid_in[i]),
                .lr_d_data_in(lr_d_data_in[i]),
                .lr_d_H_data_in(lr_d_H_in[i]),
                .lr_leak_factor_in(lr_leak_factor_in),
                .lr_d_data_out(lr_d_data_out[i]),
                .lr_d_valid_out(lr_d_valid_out[i])
            );
        end
    endgenerate

endmodule