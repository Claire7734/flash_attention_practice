#pragma once

#include "common.h"
#include "utils.h"

namespace flash {

#define BANKS_PER_VEC4_ACCESS 8
#define ELEMS_PER_BANK 8
 
__forceinline__ __device__ constexpr int get_swizzled_col(const int &row, const int &col) {
    // Restrict the swizzled column to the
    // (8, 128) byte region it's in.
    // Not strictly necessary, but we'll need it in later kernels.
    const int region_row = row % BANKS_PER_VEC4_ACCESS;
 
    // Convert column byte offset to 16B bank index since we have 8 banks of 16B each.
    // This transforms the column coordinate from element space to bank space
    const int bank_col = col / ELEMS_PER_BANK;
    
    // Preserve the byte offset within each 16B bank for non-vectorized RF→SMEM stores
    // This ensures threads in the same 4-thread group maintain their relative positions
    const int bank_offset = col % ELEMS_PER_BANK;
 
    // Apply XOR swizzling to distribute consecutive row accesses across different banks
    // Then reconstruct the final column address by scaling back to element space
    return ((region_row ^ bank_col) * ELEMS_PER_BANK) + bank_offset;
}

template <int col_fragments>
FA_DEVICE_CONSTEXPR int swizzled_col_fragment(int row, int col_fragment) {
    static_assert(col_fragments % ELEMS_PER_VEC4_ACCESS == 0,
                  "# col tiles is a multiple of # elems");

    // The % ELEMS_PER_VEC4_ACCESS makes sure that the swizzled column stays
    // within the same 8 element window.
    return (row % ELEMS_PER_VEC4_ACCESS) ^ col_fragment;
}

template <int col_fragments, bool swizzle>
FA_DEVICE_CONSTEXPR int get_smem_col_fragment(const int row,
                                              const int col_fragment) {
    return swizzle ? swizzled_col_fragment<col_fragments>(row, col_fragment)
                   : col_fragment;
}

template <const int col_fragments, const bool swizzled>
FA_DEVICE_CONSTEXPR int get_smem_offset(const int row, const int col) {
    const int offset = row * col_fragments + col;
    if constexpr (swizzled) {
        return swizzle_cute<col_fragments>(offset);
    } else {
        return offset;
    }
}


} // namespace flash
