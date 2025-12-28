#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

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
#define ARRAY_SIZE 4   // 4x4 Systolic Array
#define MATRIX_K 8     // Depth dimension (Shared dimension), must be even for packing

// -------------------------------------------------------------------------
// PE Structure
// -------------------------------------------------------------------------
typedef struct {
    int32_t  accumulator;
    uint16_t input_latch;   // Holds data from Left
    uint16_t weight_latch;  // Holds data from Top
} PE_t;

int main() {
    printf("--- Large Systolic Packed INT8 Matrix Multiplication Simulation ---\n");
    printf("--- Array Size: %dx%d, Matrix Depth: %d ---\n\n", ARRAY_SIZE, ARRAY_SIZE, MATRIX_K);
    
    srand(time(NULL));

    // ---------------------------------------------------------------------
    // 1. Define Random INT8 Matrices
    //    A: 4 Rows x 8 Cols
    //    B: 8 Rows x 4 Cols (Transposed view for initialization logic)
    // ---------------------------------------------------------------------
    int8_t A[ARRAY_SIZE][MATRIX_K];
    int8_t B[MATRIX_K][ARRAY_SIZE];

    // Initialize A with random small values (-10 to 10 to avoid huge overflows in print)
    for(int r=0; r<ARRAY_SIZE; r++) {
        for(int k=0; k<MATRIX_K; k++) {
            A[r][k] = (rand() % 21) - 10;
        }
    }

    // Initialize B
    for(int k=0; k<MATRIX_K; k++) {
        for(int c=0; c<ARRAY_SIZE; c++) {
            B[k][c] = (rand() % 21) - 10;
        }
    }

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
            A_packed[r][p] = (uint16_t)(((uint8_t)A[r][k+1] << 8) | (uint8_t)A[r][k]);
        }
    }

    // Pack B: [k+1 | k] (Vertical packing along rows of B)
    for(int c=0; c<ARRAY_SIZE; c++) {
        for(int p=0; p<packed_depth; p++) {
            int k = p * 2;
            B_packed[p][c] = (uint16_t)(((uint8_t)B[k+1][c] << 8) | (uint8_t)B[k][c]);
        }
    }


    // ---------------------------------------------------------------------
    // 4. Systolic Simulation Loop
    // ---------------------------------------------------------------------
    PE_t PEs[ARRAY_SIZE][ARRAY_SIZE];
    memset(PEs, 0, sizeof(PEs));

    // Pipeline Registers to hold "next" state logic (simulating clock edge)
    uint16_t next_input_bus[ARRAY_SIZE][ARRAY_SIZE];
    uint16_t next_weight_bus[ARRAY_SIZE][ARRAY_SIZE];

    // Number of cycles needed:
    // Skew (ARRAY_SIZE-1) + Packed Depth + Drain (ARRAY_SIZE-1) + Flush (1)
    // 3 + 4 + 3 + 1 = 11 cycles.
    int max_cycles = (ARRAY_SIZE - 1) + packed_depth + (ARRAY_SIZE - 1) + 2; 

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        // A. EXECUTE PHASE
        for(int r=0; r<ARRAY_SIZE; r++) {
            for(int c=0; c<ARRAY_SIZE; c++) {
                int32_t partial = packed_swar_dot_product(PEs[r][c].input_latch, PEs[r][c].weight_latch);
                PEs[r][c].accumulator += partial;
            }
        }

        // B. ROUTE PHASE (Calculate inputs for Next Cycle) 
        
        // 1. Internal Propagation
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
        // Row 0 starts at Cycle 0. Row R starts at Cycle R.
        for (int r=0; r<ARRAY_SIZE; r++) {
            int fetch_idx = cycle - r; // Skew subtraction
            if (fetch_idx >= 0 && fetch_idx < packed_depth) {
                next_input_bus[r][0] = A_packed[r][fetch_idx];
            } else {
                next_input_bus[r][0] = 0; // Bubble
            }
        }

        // Feed Top Side (Matrix B)
        // Col 0 starts at Cycle 0. Col C starts at Cycle C.
        for (int c=0; c<ARRAY_SIZE; c++) {
            int fetch_idx = cycle - c; // Skew subtraction
            if (fetch_idx >= 0 && fetch_idx < packed_depth) {
                next_weight_bus[0][c] = B_packed[fetch_idx][c];
            } else {
                next_weight_bus[0][c] = 0; // Bubble
            }
        }

        // C. LATCH PHASE
        for(int r=0; r<ARRAY_SIZE; r++) {
            for(int c=0; c<ARRAY_SIZE; c++) {
                PEs[r][c].input_latch  = next_input_bus[r][c];
                PEs[r][c].weight_latch = next_weight_bus[r][c];
            }
        }
    }

    // ---------------------------------------------------------------------
    // 5. Verification
    // ---------------------------------------------------------------------
    printf("--- Verification ---\n");
    int pass = 1;
    for(int r=0; r<ARRAY_SIZE; r++) {
        for(int c=0; c<ARRAY_SIZE; c++) {
            if (PEs[r][c].accumulator != C_expected[r][c]) {
                printf("PE[%d][%d]: Got %d, Expected %d [FAIL]\n", r, c, PEs[r][c].accumulator, C_expected[r][c]);
                pass = 0;
            }
        }
    }

    if(pass) {
        printf("SUCCESS: All %d PEs matched the expected Matrix Multiplication results.\n", ARRAY_SIZE*ARRAY_SIZE);
        // Print result matrix
        printf("\nResult Matrix C:\n");
        for(int r=0; r<ARRAY_SIZE; r++) {
            printf("[ ");
            for(int c=0; c<ARRAY_SIZE; c++) {
                printf("%4d ", PEs[r][c].accumulator);
            }
            printf("]\n");
        }
    } else {
        printf("\nFAILURE: Mismatch detected.\n");
    }

    return 0;
}
