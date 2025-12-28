#include <stdio.h>
#include <stdint.h>
#include <string.h>

// -------------------------------------------------------------------------
// Helper: The Packed SWAR Dot Product
// Performs: (a_high * b_high) + (a_low * b_low)
// Returns int32_t to safely hold the sum of products.
// -------------------------------------------------------------------------
int32_t packed_swar_dot_product(uint16_t input_a, uint16_t input_b) {
    int8_t a_high = (int8_t)((input_a >> 8) & 0xFF);
    int8_t a_low  = (int8_t)(input_a & 0xFF);
    int8_t b_high = (int8_t)((input_b >> 8) & 0xFF);
    int8_t b_low  = (int8_t)(input_b & 0xFF);

    int16_t prod1 = (int16_t)a_high * (int16_t)b_high;
    int16_t prod2 = (int16_t)a_low  * (int16_t)b_low;

    return (int32_t)prod1 + (int32_t)prod2;
}

// -------------------------------------------------------------------------
// Simulation Parameters
// -------------------------------------------------------------------------
#define ARRAY_SIZE 2
#define MATRIX_K 4  // Depth dimension (Shared dimension)

// -------------------------------------------------------------------------
// PE Structure
// -------------------------------------------------------------------------
typedef struct {
    int32_t  accumulator;
    uint16_t input_latch;   // Holds data from Left
    uint16_t weight_latch;  // Holds data from Top
} PE_t;

int main() {
    printf("--- Systolic Packed INT8 Matrix Multiplication Simulation ---\\n\\n");

    // ---------------------------------------------------------------------
    // 1. Define Raw INT8 Matrices
    //    A: 2 Rows x 4 Cols
    //    B: 4 Rows x 2 Cols (Transposed view for easier C initialization)
    // ---------------------------------------------------------------------
    int8_t A[ARRAY_SIZE][MATRIX_K] = {
        {1,  2,  3,  4},   // Row 0
        {5, -6,  7, -8}    // Row 1 (Mixed signs)
    };

    // B is defined as Columns for clarity (since Weights enter from Top)
    // Col 0: [1, 2, 3, 4]
    // Col 1: [5, 6, 7, 8]
    int8_t B[MATRIX_K][ARRAY_SIZE] = {
        {1, 5}, // Row 0
        {2, 6}, // Row 1
        {3, 7}, // Row 2
        {4, 8}  // Row 3
    };

    // ---------------------------------------------------------------------
    // 2. Compute Expected Result (Standard MatMul)
    // ---------------------------------------------------------------------
    int32_t C_expected[ARRAY_SIZE][ARRAY_SIZE] = {0};
    for(int r=0; r<ARRAY_SIZE; r++) {
        for(int c=0; c<ARRAY_SIZE; c++) {
            for(int k=0; k<MATRIX_K; k++) {
                C_expected[r][c] += A[r][k] * B[k][c];
            }
        }
    }

    printf("Expected Result Matrix C (Standard MatMul):\n");
    printf("[%d, %d]\n", C_expected[0][0], C_expected[0][1]);
    printf("[%d, %d]\n\\n", C_expected[1][0], C_expected[1][1]);


    // ---------------------------------------------------------------------
    // 3. Pack Data for Systolic Array
    //    Packing Factor = 2 (INT8 -> UINT16)
    // ---------------------------------------------------------------------
    int packed_depth = MATRIX_K / 2;
    uint16_t A_packed[ARRAY_SIZE][packed_depth];
    uint16_t B_packed[packed_depth][ARRAY_SIZE]; // Indexed by [PackedRow][Col]

    // Pack A: [k+1 | k]
    for(int r=0; r<ARRAY_SIZE; r++) {
        for(int p=0; p<packed_depth; p++) {
            int k = p * 2;
            // Pack: High Byte = A[r][k+1], Low Byte = A[r][k]
            A_packed[r][p] = (uint16_t)(((uint8_t)A[r][k+1] << 8) | (uint8_t)A[r][k]);
        }
    }

    // Pack B: [k+1 | k] (Vertical packing along rows of B)
    for(int c=0; c<ARRAY_SIZE; c++) {
        for(int p=0; p<packed_depth; p++) {
            int k = p * 2;
            // Pack: High Byte = B[k+1][c], Low Byte = B[k][c]
            B_packed[p][c] = (uint16_t)(((uint8_t)B[k+1][c] << 8) | (uint8_t)B[k][c]);
        }
    }

    printf("Packed Input Matrix A (Hex UINT16):\n");
    printf("[0x%04X, 0x%04X]\n", A_packed[0][0], A_packed[0][1]);
    printf("[0x%04X, 0x%04X]\n\\n", A_packed[1][0], A_packed[1][1]);


    // ---------------------------------------------------------------------
    // 4. Systolic Simulation Loop
    // ---------------------------------------------------------------------
    PE_t PEs[ARRAY_SIZE][ARRAY_SIZE];
    memset(PEs, 0, sizeof(PEs));

    // Pipeline Registers to hold "next" state logic (simulating clock edge)
    uint16_t next_input_bus[ARRAY_SIZE][ARRAY_SIZE];
    uint16_t next_weight_bus[ARRAY_SIZE][ARRAY_SIZE];

    // Number of cycles needed:
    // Skew (ARRAY_SIZE-1) + Packed Depth + Drain (ARRAY_SIZE-1)
    // Here: 1 + 2 + 1 = 4 cycles minimum to flow through, maybe +1 for flush.
    // Let's run for 6 cycles to be safe.
    int max_cycles = 6; 

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        printf("--- Cycle %d ---\\n", cycle);

        // A. EXECUTE PHASE (All PEs compute simultaneously)
        for(int r=0; r<ARRAY_SIZE; r++) {
            for(int c=0; c<ARRAY_SIZE; c++) {
                // Perform the SWAR Dot Product
                int32_t partial = packed_swar_dot_product(PEs[r][c].input_latch, PEs[r][c].weight_latch);
                
                // Accumulate
                PEs[r][c].accumulator += partial;

                // Debug Output
                if (PEs[r][c].input_latch != 0 || PEs[r][c].weight_latch != 0) {
                    printf("  PE[%d][%d]: In=0x%04X Wgt=0x%04X -> Add %d (Acc: %d)\\n", 
                        r, c, PEs[r][c].input_latch, PEs[r][c].weight_latch, partial, PEs[r][c].accumulator);
                }
            }
        }

        // B. ROUTE PHASE (Calculate inputs for Next Cycle)
        
        // 1. Internal Propagation (Left->Right, Top->Down)
        for(int r=0; r<ARRAY_SIZE; r++) {
            for(int c=0; c<ARRAY_SIZE; c++) {
                // Pass Input to the Right
                if (c < ARRAY_SIZE - 1) {
                    next_input_bus[r][c+1] = PEs[r][c].input_latch;
                }
                // Pass Weight Down
                if (r < ARRAY_SIZE - 1) {
                    next_weight_bus[r+1][c] = PEs[r][c].weight_latch;
                }
            }
        }

        // 2. Feed New Data at Boundaries (With Skew)
        
        // Feed Left Side (Matrix A)
        // Row 0 starts at Cycle 0. Row 1 starts at Cycle 1.
        for (int r=0; r<ARRAY_SIZE; r++) {
            int fetch_idx = cycle - r; // Skew subtraction
            if (fetch_idx >= 0 && fetch_idx < packed_depth) {
                next_input_bus[r][0] = A_packed[r][fetch_idx];
            } else {
                next_input_bus[r][0] = 0; // Bubble / Done
            }
        }

        // Feed Top Side (Matrix B)
        // Col 0 starts at Cycle 0. Col 1 starts at Cycle 1.
        for (int c=0; c<ARRAY_SIZE; c++) {
            int fetch_idx = cycle - c; // Skew subtraction
            if (fetch_idx >= 0 && fetch_idx < packed_depth) {
                next_weight_bus[0][c] = B_packed[fetch_idx][c];
            } else {
                next_weight_bus[0][c] = 0; // Bubble / Done
            }
        }

        // C. LATCH PHASE (Clock Edge updates registers)
        for(int r=0; r<ARRAY_SIZE; r++) {
            for(int c=0; c<ARRAY_SIZE; c++) {
                PEs[r][c].input_latch  = next_input_bus[r][c];
                PEs[r][c].weight_latch = next_weight_bus[r][c];
            }
        }
        printf("\\n");
    }

    // ---------------------------------------------------------------------
    // 5. Verification
    // ---------------------------------------------------------------------
    printf("--- Final Results vs Expected ---\\n");
    int pass = 1;
    for(int r=0; r<ARRAY_SIZE; r++) {
        for(int c=0; c<ARRAY_SIZE; c++) {
            printf("PE[%d][%d]: Got %d, Expected %d", r, c, PEs[r][c].accumulator, C_expected[r][c]);
            if (PEs[r][c].accumulator != C_expected[r][c]) {
                printf(" [FAIL]\\n");
                pass = 0;
            } else {
                printf(" [PASS]\\n");
            }
        }
    }

    if(pass) printf("\\nSUCCESS: Systolic Packed Execution Matches Standard MatMul.\\n");
    else printf("\\nFAILURE: Mismatch detected.\\n");

    return 0;
}
