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
    
    logic signed [15:0] current_weight;
    
    // Mux to select weight for current cycle (supports same-cycle switching)
    assign current_weight = pe_switch_in ? weight_reg_inactive : weight_reg_active;

    // Operand slicing for Packed Modes (defined outside always_comb for simulator safety)
    logic signed [7:0] a_hi_8, a_lo_8, w_hi_8, w_lo_8;
    logic signed [3:0] a3_4, a2_4, a1_4, a0_4;
    logic signed [3:0] w3_4, w2_4, w1_4, w0_4;

    assign a_hi_8 = pe_input_in[15:8];
    assign a_lo_8 = pe_input_in[7:0];
    assign w_hi_8 = current_weight[15:8];
    assign w_lo_8 = current_weight[7:0];

    assign a3_4 = pe_input_in[15:12];
    assign a2_4 = pe_input_in[11:8];
    assign a1_4 = pe_input_in[7:4];
    assign a0_4 = pe_input_in[3:0];

    assign w3_4 = current_weight[15:12];
    assign w2_4 = current_weight[11:8];
    assign w1_4 = current_weight[7:4];
    assign w0_4 = current_weight[3:0];

    // Intermediate wires for ALU product terms (moved to module scope for simulator safety)
    logic signed [15:0] prod_hi, prod_lo;
    logic signed [7:0] p3, p2, p1, p0;

    assign prod_hi = a_hi_8 * w_hi_8;
    assign prod_lo = a_lo_8 * w_lo_8;
    
    assign p3 = a3_4 * w3_4;
    assign p2 = a2_4 * w2_4;
    assign p1 = a1_4 * w1_4;
    assign p0 = a0_4 * w0_4;

    //---------------------------------------------------------
    // Multi-Modal ALU
    //---------------------------------------------------------
    always_comb begin
        case (sys_mode)
            // Q8.8 Fixed-Point (Legacy Mode)
            // (A * W) -> 32-bit product effectively Q16.16
            2'b00: begin
                alu_out = pe_input_in * current_weight;
            end

            // INT16 Mode
            // A * W -> 32-bit integer product
            2'b01: begin
                alu_out = pe_input_in * current_weight;
            end

            // INT8 Packed Mode (W8A8)
            // Result: (A_hi * W_hi) + (A_lo * W_lo)
            2'b10: begin
                alu_out = prod_hi + prod_lo;
            end

            // INT4 Packed Mode (W4A4)
            // Result: Sum of 4 INT4 products
            2'b11: begin
                alu_out = p3 + p2 + p1 + p0;
            end
            
            default: alu_out = '0;
        endcase

        
        // Accumulator Logic
        psum_next = alu_out + pe_psum_in;
    end

    // Only the switch flag is combinational (active register copies inactive register on the same clock cycle that switch flag is set)
    // That means inputs from the left side of the PE can load in on the same clock cycle that the switch flag is set
    // Only the switch flag is combinational (active register copies inactive register on the same clock cycle that switch flag is set)
    // That means inputs from the left side of the PE can load in on the same clock cycle that the switch flag is set
    // FIXED: Moved to always_ff to avoid multiple drivers. This effectively makes it a synchronous update.
    
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
            
            // Weight Switch Logic
            if (pe_switch_in) begin
                if (pe_accept_w_in) begin
                    weight_reg_active <= pe_weight_in;
                end else begin
                    weight_reg_active <= weight_reg_inactive;
                end
            end

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