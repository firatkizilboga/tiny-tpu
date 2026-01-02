`timescale 1ns/1ps
`default_nettype none

module pe #(
    parameter int DATA_WIDTH = 16 //TODO: remove? we're not using this yet, lol)
) (
    input logic clk,
    input logic rst,

    // North wires of PE
    input logic signed [31:0] pe_psum_in, 
    input logic signed [15:0] pe_weight_in,
    input logic pe_accept_w_in, 
    
    // West wires of PE
    input logic signed [15:0] pe_input_in, 
    input logic pe_valid_in, 
    input logic pe_switch_in, 
    input logic pe_enabled,

    // South wires of the PE
    output logic signed [31:0] pe_psum_out,
    output logic signed [15:0] pe_weight_out,

    // East wires of the PE
    output logic signed [15:0] pe_input_out,
    output logic pe_valid_out,
    output logic pe_switch_out,

    input logic [1:0] sys_mode
);

    logic signed [31:0] alu_out;
    logic signed [31:0] psum_next;
    
    logic signed [15:0] weight_reg_active; // foreground register
    logic signed [15:0] weight_reg_inactive; // background register
    
    //---------------------------------------------------------
    // Multi-Modal ALU
    //---------------------------------------------------------
    always_comb begin
        case (sys_mode)
            // Q8.8 Fixed-Point (Legacy Mode)
            // (A * W) -> 32-bit product effectively Q16.16
            2'b00: begin
                alu_out = pe_input_in * weight_reg_active;
            end

            // INT16 Mode
            // A * W -> 32-bit integer product
            2'b01: begin
                alu_out = pe_input_in * weight_reg_active;
            end

            // INT8 Packed Mode (W8A8)
            // Input:  [A_hi (8) | A_lo (8)]
            // Weight: [W_hi (8) | W_lo (8)]
            // Result: (A_hi * W_hi) + (A_lo * W_lo)
            2'b10: begin
                logic signed [7:0] a_hi, a_lo, w_hi, w_lo;
                logic signed [15:0] prod_hi, prod_lo;
                
                a_hi = pe_input_in[15:8];
                a_lo = pe_input_in[7:0];
                w_hi = weight_reg_active[15:8];
                w_lo = weight_reg_active[7:0];
                
                prod_hi = a_hi * w_hi;
                prod_lo = a_lo * w_lo;
                
                alu_out = {{16{prod_hi[15]}}, prod_hi} + {{16{prod_lo[15]}}, prod_lo};
            end

            // INT4 Packed Mode (W4A4)
            // Input:  [A3 (4) | A2 (4) | A1 (4) | A0 (4)]
            // Weight: [W3 (4) | W2 (4) | W1 (4) | W0 (4)]
            // Result: Sum of 4 INT4 products
            2'b11: begin
                logic signed [3:0] a3, a2, a1, a0;
                logic signed [3:0] w3, w2, w1, w0;
                logic signed [7:0] p3, p2, p1, p0;
                
                a3 = pe_input_in[15:12];
                a2 = pe_input_in[11:8];
                a1 = pe_input_in[7:4];
                a0 = pe_input_in[3:0];
                
                w3 = weight_reg_active[15:12];
                w2 = weight_reg_active[11:8];
                w1 = weight_reg_active[7:4];
                w0 = weight_reg_active[3:0];
                
                // Sign-extend 4-bit operands to calculate product safely
                p3 = a3 * w3;
                p2 = a2 * w2;
                p1 = a1 * w1;
                p0 = a0 * w0;
                
                alu_out = {{24{p3[7]}}, p3} + 
                          {{24{p2[7]}}, p2} + 
                          {{24{p1[7]}}, p1} + 
                          {{24{p0[7]}}, p0};
            end
            
            default: alu_out = '0;
        endcase
        
        // Accumulator Logic
        psum_next = alu_out + pe_psum_in;
    end

    // Only the switch flag is combinational (active register copies inactive register on the same clock cycle that switch flag is set)
    // That means inputs from the left side of the PE can load in on the same clock cycle that the switch flag is set
    always_comb begin
        if (pe_switch_in) begin
            weight_reg_active = weight_reg_inactive;
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst || !pe_enabled) begin
            pe_input_out <= 16'b0;
            weight_reg_active <= 16'b0;
            weight_reg_inactive <= 16'b0;
            pe_valid_out <= 0;
            pe_weight_out <= 16'b0;
            pe_switch_out <= 0;
        end else begin
            pe_valid_out <= pe_valid_in;
            pe_switch_out <= pe_switch_in;
            
            // Weight register updates - only on clock edges
            if (pe_accept_w_in) begin
                weight_reg_inactive <= pe_weight_in;
                pe_weight_out <= pe_weight_in;
            end else begin
                pe_weight_out <= 0;
            end

            if (pe_valid_in) begin
                pe_input_out <= pe_input_in;
                pe_psum_out <= psum_next;
            end else begin
                pe_valid_out <= 0;
                pe_psum_out <= 32'b0;
            end

        end
    end

endmodule