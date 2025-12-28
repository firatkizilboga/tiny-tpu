#include <stdio.h>
#include <stdint.h> // For int8_t, int16_t, uint16_t, int32_t

// Function to perform a packed SWAR dot product on two 16-bit words
// Each 16-bit word is treated as two packed 8-bit signed integers.
// The operation is (a_high * b_high) + (a_low * b_low).
int32_t packed_swar_dot_product(uint16_t input_a, uint16_t input_b) {
    // Extract individual 8-bit signed integers
    // Cast to int8_t to ensure signed interpretation
    int8_t a_high = (int8_t)((input_a >> 8) & 0xFF);
    int8_t a_low  = (int8_t)(input_a & 0xFF);
    
    int8_t b_high = (int8_t)((input_b >> 8) & 0xFF);
    int8_t b_low  = (int8_t)(input_b & 0xFF);

    // Perform multiplications. The result of int8_t * int8_t can be int16_t.
    int16_t prod1 = (int16_t)a_high * (int16_t)b_high;
    int16_t prod2 = (int16_t)a_low  * (int16_t)b_low;

    // Sum the products. The sum of two int16_t can exceed int16_t range,
    // so use int32_t for the final result.
    int32_t result = (int32_t)prod1 + (int32_t)prod2;

    return result;
}

int main() {
    // Example 1: Positive values
    // Input A: [10 | 5] -> uint16_t: (10 << 8) | 5 = 0x0A05
    // Input B: [ 4 | 2] -> uint16_t: ( 4 << 8) | 2 = 0x0402
    // Expected: (10 * 4) + (5 * 2) = 40 + 10 = 50
    uint16_t a1 = (uint16_t)((10 << 8) | 5);
    uint16_t b1 = (uint16_t)(( 4 << 8) | 2);
    int32_t res1 = packed_swar_dot_product(a1, b1);
    printf("Example 1:\n");
    printf("  Input A: 0x%04X (High: %d, Low: %d)\n", a1, (int8_t)((a1 >> 8) & 0xFF), (int8_t)(a1 & 0xFF));
    printf("  Input B: 0x%04X (High: %d, Low: %d)\n", b1, (int8_t)((b1 >> 8) & 0xFF), (int8_t)(b1 & 0xFF));
    printf("  Result : %d (Expected: 50)\n\n", res1);

    // Example 2: Mixed positive and negative values
    // Input A: [-10 | 5] -> uint16_t: (-10 << 8) | 5 = 0xF605 (two's complement for -10 is 0xF6)
    // Input B: [  4 | -2] -> uint16_t: ( 4 << 8) | -2 = 0x04FE (two's complement for -2 is 0xFE)
    // Expected: (-10 * 4) + (5 * -2) = -40 + -10 = -50
    uint16_t a2 = (uint16_t)((((uint8_t)-10) << 8) | ((uint8_t)5));
    uint16_t b2 = (uint16_t)(((uint8_t)4 << 8) | ((uint8_t)-2));
    int32_t res2 = packed_swar_dot_product(a2, b2);
    printf("Example 2:\n");
    printf("  Input A: 0x%04X (High: %d, Low: %d)\n", a2, (int8_t)((a2 >> 8) & 0xFF), (int8_t)(a2 & 0xFF));
    printf("  Input B: 0x%04X (High: %d, Low: %d)\n", b2, (int8_t)((b2 >> 8) & 0xFF), (int8_t)(b2 & 0xFF));
    printf("  Result : %d (Expected: -50)\n\n", res2);

    // Example 3: Larger values, near limits
    // Input A: [100 | 80]
    // Input B: [  5 | 10]
    // Expected: (100 * 5) + (80 * 10) = 500 + 800 = 1300
    uint16_t a3 = (uint16_t)((100 << 8) | 80);
    uint16_t b3 = (uint16_t)((  5 << 8) | 10);
    int32_t res3 = packed_swar_dot_product(a3, b3);
    printf("Example 3:\n");
    printf("  Input A: 0x%04X (High: %d, Low: %d)\n", a3, (int8_t)((a3 >> 8) & 0xFF), (int8_t)(a3 & 0xFF));
    printf("  Input B: 0x%04X (High: %d, Low: %d)\n", b3, (int8_t)((b3 >> 8) & 0xFF), (int8_t)(b3 & 0xFF));
    printf("  Result : %d (Expected: 1300)\n\n", res3);

    // Example 4: Max accumulation possible for two products (int8_t range -128 to 127)
    // Max single product: 127 * 127 = 16129
    // Max sum: 16129 + 16129 = 32258 (fits in int16_t, but int32_t is safer for general use)
    uint16_t a4 = (uint16_t)((127 << 8) | 127);
    uint16_t b4 = (uint16_t)((127 << 8) | 127);
    int32_t res4 = packed_swar_dot_product(a4, b4);
    printf("Example 4 (Max Positive):\n");
    printf("  Input A: 0x%04X (High: %d, Low: %d)\n", a4, (int8_t)((a4 >> 8) & 0xFF), (int8_t)(a4 & 0xFF));
    printf("  Input B: 0x%04X (High: %d, Low: %d)\n", b4, (int8_t)((b4 >> 8) & 0xFF), (int8_t)(b4 & 0xFF));
    printf("  Result : %d (Expected: 32258)\n\n", res4);

    // Example 5: Min accumulation possible for two products (int8_t range -128 to 127)
    // Min single product: -128 * 127 = -16256
    // Min sum: -16256 + -16256 = -32512 (fits in int16_t, but int32_t is safer)
    uint16_t a5 = (uint16_t)((((uint8_t)-128) << 8) | ((uint8_t)-128));
    uint16_t b5 = (uint16_t)(((uint8_t)127 << 8) | ((uint8_t)127));
    int32_t res5 = packed_swar_dot_product(a5, b5);
    printf("Example 5 (Min Negative):\n");
    printf("  Input A: 0x%04X (High: %d, Low: %d)\n", a5, (int8_t)((a5 >> 8) & 0xFF), (int8_t)(a5 & 0xFF));
    printf("  Input B: 0x%04X (High: %d, Low: %d)\n", b5, (int8_t)((b5 >> 8) & 0xFF), (int8_t)(b5 & 0xFF));
    printf("  Result : %d (Expected: -32512)\n\n", res5);

    return 0;
}
