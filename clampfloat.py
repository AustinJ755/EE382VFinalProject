import struct
import numpy as np
from cffi import FFI

# C code version of the clamp float function for speedup
ffi = FFI()
ffi.set_source("_clampfloat", """
#include <immintrin.h>
void clamp_float(unsigned long long ptr, unsigned n, unsigned s_bits, unsigned e_bits, unsigned m_bits) {
    //printf("test1\\n");
    float * values = (float *) ptr;
    unsigned man_size_diff = 23 - m_bits;
    int max_exp = (1 << (e_bits-1))-1;
    

    int maskvalue = (0x7FFFFF>>man_size_diff)<<man_size_diff;
    //value to mask mantissa
    __m256i mantMask=_mm256_set1_epi32(maskvalue);
    //bias
    __m256i val127=_mm256_set1_epi32(127);
    //max allowed exponent
    __m256i maxEXP = _mm256_set1_epi32(max_exp);
    //minallowed exponent
    __m256i minEXP = _mm256_set1_epi32(-max_exp);
    int i=0;
    for(;i<n-8;i+=8){
            //load the data into simd registers
            __m256i input = _mm256_load_si256((__m256i*)&values[i]);
            //Truncate the mantissa value 
            __m256i mantissa =_mm256_and_si256(input,mantMask);
        
            //Remove sign bit
            __m256i exponent = _mm256_slli_epi32(input,1);
            //remove mantissa and shift exponent to bottom
            exponent = _mm256_srli_epi32(exponent,24);

            //subtract bias
            exponent = _mm256_sub_epi32(exponent,val127);

            //clamp the exponent
            exponent = _mm256_min_epi32(exponent,maxEXP);
            exponent = _mm256_max_epi32(exponent,minEXP);

            //add bias back
            exponent = _mm256_add_epi32(exponent,val127);

            //move exponent back to starting position
            exponent = _mm256_slli_epi32(exponent,23);

            //assume sign bit will always be used 
            input =  _mm256_srli_epi32(input,31);
            input = _mm256_slli_epi32(input,31);

            input = _mm256_or_si256(input,exponent);
            input = _mm256_or_si256(input,mantissa);
            
            _mm256_store_si256((__m256i*)&values[i],input);
    }
    for(; i < n; i++) {
        unsigned * value_i_ptr = (unsigned *) &(values[i]);
        unsigned value_i = *value_i_ptr;
        unsigned res_i = 0;
        float * res_f_ptr = (float *) &res_i;

        // Truncate the mantissa
        unsigned man = value_i&maskvalue;
        res_i |= man;

        // Compute the effective exponent, then clamp to the representable range
        int eff_exp = (value_i >> 23 & 0xFF) - 127;
        //int res_exp = std::min(max_exp,eff_exp);
        int res_exp = (max_exp > eff_exp) ? eff_exp : max_exp;
        res_exp = (-max_exp > eff_exp) ? -max_exp : eff_exp;
        res_exp += 127; // Add back bias
        res_i |= res_exp << 23;

        // Handle the sign
        unsigned sign = value_i & 0x80000000;
        if(s_bits == 0 && sign != 0) {
            // If unsigned, clamp negatives to zero
            res_i = 0;
        } else {
            // Otherwise, push the sign bit back in
            res_i |= sign;
        }
        values[i] = *res_f_ptr;
    }
}
""", extra_compile_args=["-mavx2","-mavx","-Ofast"])
# ffi.set_source("_clampfloat", """
# void clamp_float(unsigned long long ptr, unsigned n, unsigned s_bits, unsigned e_bits, unsigned m_bits) {
#     //printf("test1\\n");
#     float * values = (float *) ptr;
#     unsigned man_size_diff = 23 - m_bits;
#     int max_exp = (1 << (e_bits-1))-1;
#         int maskvalue = (0x7FFFFF>>man_size_diff)<<man_size_diff;
#     for(unsigned i = 0; i < n; i++) {
#         unsigned * value_i_ptr = (unsigned *) &(values[i]);
#         unsigned value_i = *value_i_ptr;
#         unsigned res_i = 0;
#         float * res_f_ptr = (float *) &res_i;

#         // Truncate the mantissa
#         unsigned man = value_i&maskvalue;
#         res_i |= man;

#         // Compute the effective exponent, then clamp to the representable range
#         int eff_exp = (value_i >> 23 & 0xFF) - 127;
        
#         int res_exp = (max_exp > eff_exp) ? eff_exp : max_exp;
#         res_exp = (-max_exp > eff_exp) ? -max_exp : eff_exp;
#         res_exp += 127; // Add back bias
#         res_i |= res_exp << 23;

#         // Handle the sign
#         unsigned sign = value_i & 0x80000000;
#         if(s_bits == 0 && sign != 0) {
#             // If unsigned, clamp negatives to zero
#             res_i = 0;
#         } else {
#             // Otherwise, push the sign bit back in
#             res_i |= sign;
#         }
#         values[i] = *res_f_ptr;
#     }
# }""", extra_compile_args=["--Ofast"])
ffi.cdef("""void clamp_float(unsigned long long, unsigned, unsigned, unsigned, unsigned);""")
ffi.compile()
from _clampfloat import lib  # import the compiled library

# Function that clamps a F32 value to a representable value given floating point parameters (in place)
# This is the faster function that calls the compiled C library
# Prints: nothing
# Returns: an array of clamped floating point values
def vec_clamp_float(values, s_bits=1, e_bits=8, m_bits=23):
    #print(values.__array_interface__['data'][0])
    lib.clamp_float(values.__array_interface__['data'][0], values.size, s_bits, e_bits, m_bits)


### Python functions (SLOW)

# Function that converts float to integer
# Prints: nothing
# Returns: an integer value that represents the binary float value
def float_to_int(value):
    [d] = struct.unpack(">L", struct.pack(">f", value))
    return d

# Function that converts integer to float
# Prints: nothing
# Returns: the floating point value represented by the integer's binary value
def int_to_float(value):
    [f] = struct.unpack(">f", struct.pack(">L", value))
    return f

# OLD Function that clamps a F32 value to a representable value given floating point parameters
# This is the slow Python implementation
# Prints: nothing
# Retunrs: a clamped floating point value
def clamp_float_python(value_f, s_bits=1, e_bits=8, m_bits=23):
    # First, convert the float to an integer for bit manipulation
    value_i = float_to_int(value_f)
    res_i = 0

    # Truncate the mantissa
    man_size_diff = 23 - m_bits
    man = (value_i & 0x7FFFFF) >> man_size_diff
    man = man << man_size_diff
    res_i |= man

    # Compute the effective exponent, then clamp to the representable range
    eff_exp = (value_i >> 23 & 0xFF) - 127
    max_exp = (1 << (e_bits-1))-1
    res_exp = min(max_exp, eff_exp)
    res_exp = max(-max_exp, res_exp)
    res_exp = res_exp + 127 # Add back bias
    res_i |= res_exp << 23

    sign = value_i & 0x80000000
    # If unsigned, clamp negatives to zero
    if s_bits == 0 and sign != 0:
        res_i = 0
    # Otherwise, push the sign bit back in
    else:
        res_i |= sign

    # Convert the integer back to float and return
    res_f = int_to_float(res_i)
    return np.single(res_f)

# Create a NumPy vectorized version of this function
# vec_clamp_float = np.vectorize(clamp_float)