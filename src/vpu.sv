`timescale 1ns/1ps
`default_nettype none

// three data pathways:
// (forward pass hidden layer computations) input from sys --> bias --> leaky relu --> output
// (transition) input from sys --> bias --> leaky relu --> loss --> leaky relu derivative --> output
// (backward pass) input from sys --> leaky relu derivative --> output
// during the transition pathway we need to store the H matrices that come out of the leaky relu modules AND pass them to the loss modules

/* 
|bias control bit| |lr control bit| |loss control bit| |lr_d control bit|

0000: activate no modules
1100: forward pass pathway (sys --> bias --> leaky relu --> output)
1111: transistion pathway (sys --> bias --> leaky relu --> loss --> leaky relu derivative --> output)
0001: backward pass pathway (sys --> leaky relu derivative --> output)
*/

module vpu #(
    parameter int N = 2
)(
    input logic clk,
    input logic rst,

    input logic [3:0] vpu_data_pathway, // 4-bits to signify which modules to route the inputs to (1 bit for each module)
    input logic [1:0] sys_mode,

    // Requantization parameters
    input logic [15:0] requant_scale,
    input logic [15:0] requant_shift,
    input logic [15:0] requant_zero_point,

    // Inputs from systolic array
    input logic signed [N-1:0][31:0] vpu_data_in,
    input logic [N-1:0] vpu_valid_in,

    // Inputs from UB
    input logic signed [N-1:0][15:0] bias_scalar_in,        // For bias modules
    input logic signed [15:0] lr_leak_factor_in,           // For leaky relu modules
    input logic signed [N-1:0][15:0] Y_in,                 // For loss modules
    input logic signed [15:0] inv_batch_size_times_two_in, // For loss modules
    input logic signed [N-1:0][15:0] H_in,                 // For leaky relu derivative modules 
    
    // Outputs to UB
    output logic signed [N-1:0][15:0] vpu_data_out,
    output logic [N-1:0] vpu_valid_out
);

    // Requantized inputs (from 32-bit sys array output to 16-bit VPU pipeline)
    logic signed [N-1:0][15:0] requant_out;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_requant
            assign requant_out[i] = (sys_mode == 2'b00) ? (vpu_data_in[i] >>> 8) : vpu_data_in[i][15:0];
        end
    endgenerate

    // bias
    logic signed [N-1:0][15:0] bias_data_in; 
    logic [N-1:0] bias_valid_in;
    logic signed [N-1:0][15:0] bias_z_data_out;
    logic [N-1:0] bias_valid_out;

    // bias to lr intermediate values
    logic signed [N-1:0][15:0] b_to_lr_data_in;
    logic [N-1:0] b_to_lr_valid_in;

    // lr
    logic signed [N-1:0][15:0] lr_data_in; 
    logic [N-1:0] lr_valid_in;
    logic signed [N-1:0][15:0] lr_data_out;
    logic [N-1:0] lr_valid_out;

    // lr to loss intermediate values
    logic signed [N-1:0][15:0] lr_to_loss_data_in;
    logic [N-1:0] lr_to_loss_valid_in;

    // loss
    logic signed [N-1:0][15:0] loss_data_in; 
    logic [N-1:0] loss_valid_in;
    logic signed [N-1:0][15:0] loss_data_out;
    logic [N-1:0] loss_valid_out;

    // loss to lrd intermediate values
    logic signed [N-1:0][15:0] loss_to_lrd_data_in;
    logic [N-1:0] loss_to_lrd_valid_in;

    // lr_d
    logic signed [N-1:0][15:0] lr_d_data_in; 
    logic [N-1:0] lr_d_valid_in;
    logic signed [N-1:0][15:0] lr_d_data_out;
    logic [N-1:0] lr_d_valid_out;
    logic signed [N-1:0][15:0] lr_d_H_in;

    // temp 'last H matrix' cache
    logic signed [N-1:0][15:0] last_H_data_in;
    logic signed [N-1:0][15:0] last_H_data_out;

    // Instantiate Child Modules in Parallel Loops
    generate
        for (i=0; i<N; i++) begin : bias_gen
            bias_child bias_inst (  
                .clk(clk),
                .rst(rst),
                .bias_sys_data_in(bias_data_in[i]),
                .bias_sys_valid_in(bias_valid_in[i]),
                .bias_scalar_in(bias_scalar_in[i]),
                .bias_Z_valid_out(bias_valid_out[i]),
                .bias_z_data_out(bias_z_data_out[i])
            );
        end

        for (i=0; i<N; i++) begin : lr_gen
            leaky_relu_child leaky_relu_inst (
                .clk(clk),
                .rst(rst),
                .lr_data_in(lr_data_in[i]),
                .lr_valid_in(lr_valid_in[i]),
                .lr_leak_factor_in(lr_leak_factor_in), // Shared
                .lr_data_out(lr_data_out[i]), 
                .lr_valid_out(lr_valid_out[i])
            );
        end

        for (i=0; i<N; i++) begin : loss_gen
            loss_child loss_inst (
                .clk(clk),
                .rst(rst),
                .H_in(loss_data_in[i]),
                .Y_in(Y_in[i]),
                .valid_in(loss_valid_in[i]),
                .inv_batch_size_times_two_in(inv_batch_size_times_two_in), // Shared
                .gradient_out(loss_data_out[i]),
                .valid_out(loss_valid_out[i])
            );
        end

        for (i=0; i<N; i++) begin : lrd_gen
            leaky_relu_derivative_child leaky_relu_derivative_inst (
                .clk(clk),
                .rst(rst),
                .lr_d_data_in(lr_d_data_in[i]),
                .lr_d_valid_in(lr_d_valid_in[i]),
                .lr_d_H_data_in(lr_d_H_in[i]),
                .lr_leak_factor_in(lr_leak_factor_in), // Shared
                .lr_d_data_out(lr_d_data_out[i]),
                .lr_d_valid_out(lr_d_valid_out[i])
            );
        end
    endgenerate

    always @(*) begin
        if (rst) begin
            vpu_data_out = '0;
            vpu_valid_out = '0;
            
            // default internal wire assignments during reset
            bias_data_in = '0;
            bias_valid_in = '0;
            lr_data_in = '0;
            lr_valid_in = '0;
            loss_data_in = '0;
            loss_valid_in = '0;
            lr_d_data_in = '0;
            lr_d_valid_in = '0;
        end else begin
            // bias module
            if(vpu_data_pathway[3]) begin
                // connect vpu inputs to bias module
                bias_data_in = requant_out;
                bias_valid_in = vpu_valid_in;

                // connect bias output to intermediate values
                b_to_lr_data_in = bias_z_data_out;
                b_to_lr_valid_in = bias_valid_out;
            end else begin
                // disable inputs
                bias_data_in = '0;
                bias_valid_in = '0;

                // connect vpu input to intermediate values
                b_to_lr_data_in = requant_out;
                b_to_lr_valid_in = vpu_valid_in;
            end

            // leaky relu module
            if(vpu_data_pathway[2]) begin
                // connect lr inputs to intermediate values
                lr_data_in = b_to_lr_data_in;
                lr_valid_in = b_to_lr_valid_in;

                // connect lr outputs to intermediate values
                lr_to_loss_data_in = lr_data_out;
                lr_to_loss_valid_in = lr_valid_out;
            end else begin
                // disable inputs
                lr_data_in = '0;
                lr_valid_in = '0;

                // connect intermediate values to each other
                lr_to_loss_data_in = b_to_lr_data_in;
                lr_to_loss_valid_in = b_to_lr_valid_in;
            end

            // loss module
            if(vpu_data_pathway[1]) begin
                // connect loss inputs to intermediate values
                loss_data_in = lr_to_loss_data_in;
                loss_valid_in = lr_to_loss_valid_in;

                // connect loss outputs to intermediate values
                loss_to_lrd_data_in = loss_data_out;
                loss_to_lrd_valid_in = loss_valid_out;

                // Cache and use 'last H matrix'
                last_H_data_in = lr_data_out;
                lr_d_H_in = last_H_data_out;
            end else begin
                // disable inputs
                loss_data_in = '0;
                loss_valid_in = '0;

                // connect intermediate values to each other
                loss_to_lrd_data_in = lr_to_loss_data_in;
                loss_to_lrd_valid_in = lr_to_loss_valid_in;

                // Don't cache and use 'last H matrix'
                lr_d_H_in = H_in;
            end

            // leaky relu derivative module
            if(vpu_data_pathway[0]) begin
                lr_d_data_in = loss_to_lrd_data_in;
                lr_d_valid_in = loss_to_lrd_valid_in;

                // connect lr_d outputs to vpu output
                vpu_data_out = lr_d_data_out;
                vpu_valid_out = lr_d_valid_out;
            end else begin
                // disable inputs
                lr_d_data_in = loss_to_lrd_data_in;
                lr_d_valid_in = loss_to_lrd_valid_in;

                // connect intermediate values to vpu output
                vpu_data_out = loss_to_lrd_data_in;
                vpu_valid_out = loss_to_lrd_valid_in;
            end
        end
    end

    // sequential logic to cache last H
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            last_H_data_out <= '0;
        end else begin
            if (vpu_data_pathway[1]) begin
                last_H_data_out <= last_H_data_in;
            end else begin
                last_H_data_out <= '0;
            end 
        end
    end

endmodule