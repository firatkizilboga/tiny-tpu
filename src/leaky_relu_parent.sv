`timescale 1ns/1ps
`default_nettype none

module leaky_relu_parent #(
    parameter int N = 2
)(
    input logic clk,
    input logic rst,
    input logic signed [15:0] lr_leak_factor_in,

    input logic [N-1:0] lr_valid_in,

    input logic signed [N-1:0][15:0] lr_data_in,
    
    output logic signed [N-1:0][15:0] lr_data_out,
    
    output logic [N-1:0] lr_valid_out
);

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : leaky_relu_cols
            leaky_relu_child leaky_relu_col (
                .clk(clk),
                .rst(rst),
                .lr_valid_in(lr_valid_in[i]),
                .lr_data_in(lr_data_in[i]),
                .lr_leak_factor_in(lr_leak_factor_in),
                .lr_data_out(lr_data_out[i]),
                .lr_valid_out(lr_valid_out[i])
            );
        end
    endgenerate

endmodule