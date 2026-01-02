`timescale 1ns/1ps
`default_nettype none

module loss_parent #(
    parameter int N = 2
)(
    input logic clk,
    input logic rst,
    input logic signed [N-1:0][15:0] H_in,
    input logic signed [N-1:0][15:0] Y_in,
    input logic [N-1:0] valid_in,
    input logic signed [15:0] inv_batch_size_times_two_in,
    
    output logic signed [N-1:0][15:0] gradient_out,
    output logic [N-1:0] valid_out
);

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : loss_cols
            loss_child loss_col (
                .clk(clk),
                .rst(rst),
                .H_in(H_in[i]),
                .Y_in(Y_in[i]),
                .valid_in(valid_in[i]),
                .inv_batch_size_times_two_in(inv_batch_size_times_two_in),
                .gradient_out(gradient_out[i]),
                .valid_out(valid_out[i])
            );
        end
    endgenerate

endmodule