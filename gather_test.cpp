#include <immintrin.h>

#ifdef _MSC_VER
#include <conio.h>
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <utility>

// taken from http://stackoverflow.com/a/31981256/4699324
#ifdef _MSC_VER
#    if (_MSC_VER >= 1800)
#        define __alignas_is_defined 1
#    endif
#    if (_MSC_VER >= 1900)
#        define __alignof_is_defined 1
#    endif
#else
#    include <stdalign.h>   // __alignas/of_is_defined directly from the implementation
#endif

#ifdef __alignas_is_defined
#    define ALIGN(X) alignas(X)
#else
#    pragma message("C++11 alignas unsupported :( Falling back to compiler attributes")
#    ifdef __GNUG__
#        define ALIGN(X) __attribute__ ((aligned(X)))
#    elif defined(_MSC_VER)
#        define ALIGN(X) __declspec(align(X))
#    else
#        error Unknown compiler, unknown alignment attribute!
#    endif
#endif

#ifdef __alignof_is_defined
#    define ALIGNOF(X) alignof(x)
#else
#    pragma message("C++11 alignof unsupported :( Falling back to compiler attributes")
#    ifdef __GNUG__
#        define ALIGNOF(X) __alignof__ (X)
#    elif defined(_MSC_VER)
#        define ALIGNOF(X) __alignof(X)
#    else
#        error Unknown compiler, unknown alignment attribute!
#    endif
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

#ifdef _MSC_VER
#define YESINLINE __forceinline
#else
#define YESINLINE __attribute__((always_inline)) inline
#endif

#ifdef _MSC_VER

static YESINLINE __m256i
ymm_u8lookup_naive(const uint8_t table[256], __m256i idx)
{
    __m256i ret;
    for (int i = 0; i<32; ++i) {
        ret.m256i_u8[i] = table[idx.m256i_u8[i]];
    }
    return ret;
}

#endif

static YESINLINE __m256i
ymm_u8lookup_avx2gather(const uint8_t* lut, __m256i vindex, __m256i andMask) {

    __m256i lo = _mm256_unpacklo_epi8(vindex, _mm256_setzero_si256());
    __m256i hi = _mm256_unpackhi_epi8(vindex, _mm256_setzero_si256());
    __m256i idx0 = _mm256_unpacklo_epi16(lo, _mm256_setzero_si256());
    __m256i idx1 = _mm256_unpackhi_epi16(lo, _mm256_setzero_si256());
    __m256i idx2 = _mm256_unpacklo_epi16(hi, _mm256_setzero_si256());
    __m256i idx3 = _mm256_unpackhi_epi16(hi, _mm256_setzero_si256());

    const int* base = (const int*)lut;
    __m256i nidx0 = _mm256_i32gather_epi32(base, idx0, 1);
    __m256i nidx1 = _mm256_i32gather_epi32(base, idx1, 1);
    __m256i nidx2 = _mm256_i32gather_epi32(base, idx2, 1);
    __m256i nidx3 = _mm256_i32gather_epi32(base, idx3, 1);

    nidx0 = _mm256_and_si256(nidx0, andMask);
    nidx1 = _mm256_and_si256(nidx1, andMask);
    nidx2 = _mm256_and_si256(nidx2, andMask);
    nidx3 = _mm256_and_si256(nidx3, andMask);

    nidx0 = _mm256_packus_epi32(nidx0, nidx1);
    nidx2 = _mm256_packus_epi32(nidx2, nidx3);
    nidx0 = _mm256_packus_epi16(nidx0, nidx2);

    __m256i ret = nidx0;
    return ret;
}

template <unsigned N>
YESINLINE __m256i
ymm_u8lookup_avx2shuffle(
//  const __m128i* lut,
    const __m256i* lut,
    __m256i vindex,
    __m256i m256i_u8_all_16,
    __m256i m256i_u8_112_Mask
) {
    static_assert(N != 0, "N must not be 0.");
    static_assert(N <= 16, "N must be less than or equal to 16.");

    // a heck a lot of instructions needed...
    //LOOKUP(0)
//  __m256i t = _mm256_broadcastsi128_si256(lut[0]);
    __m256i t = _mm256_loadu_si256(lut + 0);
    __m256i tmp = _mm256_adds_epu8(vindex, m256i_u8_112_Mask);
    __m256i s = _mm256_sub_epi8(vindex, m256i_u8_all_16);
    __m256i ret = _mm256_shuffle_epi8(t, tmp);
    if (N == 1) return ret;

#define LOOKUP(idx) \
    t = _mm256_loadu_si256(lut + idx);\
    tmp = _mm256_adds_epu8(s, m256i_u8_112_Mask);\
    s = _mm256_sub_epi8(s, m256i_u8_all_16);\
    tmp = _mm256_shuffle_epi8(t, tmp);\
    ret = _mm256_or_si256(ret, tmp); \
    if (idx + 1 == N) return ret;

    LOOKUP(1)
    LOOKUP(2)
    LOOKUP(3)
    LOOKUP(4)
    LOOKUP(5)
    LOOKUP(6)
    LOOKUP(7)
    LOOKUP(8)
    LOOKUP(9)
    LOOKUP(10)
    LOOKUP(11)
    LOOKUP(12)
    LOOKUP(13)
    LOOKUP(14)
    LOOKUP(15)
#undef LOOKUP
}

// taken from https://gist.github.com/tanakamura/7c159d27f744fc24ff8243522b166820
#define UNROLL16(F)                             \
    F(0)                                        \
    F(1)                                        \
    F(2)                                        \
    F(3)                                        \
                                                \
    F(4)                                        \
    F(5)                                        \
    F(6)                                        \
    F(7)                                        \
                                                \
    F(8)                                        \
    F(9)                                        \
    F(10)                                       \
    F(11)                                       \
                                                \
    F(12)                                       \
    F(13)                                       \
    F(14)                                       \
    F(15)                                       \


#define UNROLL32(F)                             \
    F(0)                                        \
    F(1)                                        \
    F(2)                                        \
    F(3)                                        \
                                                \
    F(4)                                        \
    F(5)                                        \
    F(6)                                        \
    F(7)                                        \
                                                \
    F(8)                                        \
    F(9)                                        \
    F(10)                                       \
    F(11)                                       \
                                                \
    F(12)                                       \
    F(13)                                       \
    F(14)                                       \
    F(15)                                       \
                                                \
    F(16)                                       \
    F(17)                                       \
    F(18)                                       \
    F(19)                                       \
                                                \
    F(20)                                       \
    F(21)                                       \
    F(22)                                       \
    F(23)                                       \
                                                \
    F(24)                                       \
    F(25)                                       \
    F(26)                                       \
    F(27)                                       \
                                                \
    F(28)                                       \
    F(29)                                       \
    F(30)                                       \
    F(31)                                       \

#define UNROLL31(F)                             \
    F(0)                                        \
    F(1)                                        \
    F(2)                                        \
    F(3)                                        \
                                                \
    F(4)                                        \
    F(5)                                        \
    F(6)                                        \
    F(7)                                        \
                                                \
    F(8)                                        \
    F(9)                                        \
    F(10)                                       \
    F(11)                                       \
                                                \
    F(12)                                       \
    F(13)                                       \
    F(14)                                       \
    F(15)                                       \
                                                \
    F(16)                                       \
    F(17)                                       \
    F(18)                                       \
    F(19)                                       \
                                                \
    F(20)                                       \
    F(21)                                       \
    F(22)                                       \
    F(23)                                       \
                                                \
    F(24)                                       \
    F(25)                                       \
    F(26)                                       \
    F(27)                                       \
                                                \
    F(28)                                       \
    F(29)                                       \
    F(30)                                       \


static YESINLINE __m256i
mov32(unsigned char *table, __m256i idx)
{
    __m128i idx_lo = _mm256_extractf128_si256(idx,0);
    __m128i idx_hi = _mm256_extractf128_si256(idx,1);
    __m128i lo = _mm_undefined_si128(), hi = _mm_undefined_si128();

    unsigned char c;

#define INDEX_LO(N)                             \
    c = table[_mm_extract_epi8(idx_lo,N)];      \
    lo = _mm_insert_epi8(lo, c, N);             \


#define INDEX_HI(N)                             \
    c = table[_mm_extract_epi8(idx_hi,N)];      \
    hi = _mm_insert_epi8(hi, c, N);

    UNROLL16(INDEX_LO);
    UNROLL16(INDEX_HI);

    __m256i ret = _mm256_castsi128_si256(lo);

    return _mm256_insertf128_si256(ret, hi, 1);
}

static YESINLINE __m256i
full_scalar(unsigned char *table, __m256i idx)
{
    ALIGN(32) unsigned char idx_scalar[32];
    ALIGN(32) unsigned char result[32];

    _mm256_store_si256((__m256i*)idx_scalar, idx);

    /* result[0] = table[idx_scalar[0]];
     * result[1] = table[idx_scalar[1]];
     * ...
     * result[31] = table[idx_scalar[31]];
     */

#define SCALAR(N)\
    result[N] = table[idx_scalar[N]];

    UNROLL32(SCALAR);

    return _mm256_load_si256((__m256i*)result);
}

static NOINLINE void
test_results(const unsigned char idx[256], unsigned char val[256])
{
    const __m256i* vidx = (const __m256i*)idx;
    __m256i r;
    __m256i ref_results[8];
    __m256i results[8];

    auto match = [&](const char* name){
        char buff[32];
        bool failed = false;
        for (int i=0; i<8; ++i) {
            __m256i a = _mm256_cmpeq_epi8(ref_results[i], results[i]);
            _mm256_storeu_si256((__m256i*)buff, a);
            for (int i=0; i<32; ++i) {
                if (!buff[i]) {
                    failed = true;
                }
            }
            results[i] = _mm256_setzero_si256();
        }
        printf("%s %s\n", (failed ? "FAIL" : "PASS"), name);
    };

    uint8_t* pref = (uint8_t*) &ref_results[0];
    for (int i=0; i<256; ++i) {
        pref[i] = val[idx[i]];
    }


#ifdef _MSC_VER
    // naive
    for (int i=0; i<8; ++i) {
        r = ymm_u8lookup_naive(val, vidx[i]);
        results[i] = r;
    }
    match("naive");
#endif

    // avx2gather
    __m256i mask_FF = _mm256_set1_epi32(0xFF);
    for (int i=0; i<8; ++i) {
        r = ymm_u8lookup_avx2gather(val, vidx[i], mask_FF);
        results[i] = r;
    }
    match("avx2gather");

    // avx2shuffle
    __m256i m256i_u8_all_16 = _mm256_set1_epi8(0x10);
    __m256i m256i_u8_112_Mask = _mm256_set1_epi8(112);
    __m256i lut[32];
    for (size_t i = 0; i < 16; ++i) {
        __m128i tmp = _mm_loadu_si128((__m128i*)(val + i * 16));
        lut[i] = _mm256_broadcastsi128_si256(tmp);
    }
    for (int i=0; i<8; ++i) {
        r = ymm_u8lookup_avx2shuffle<16>(lut, vidx[i], m256i_u8_all_16, m256i_u8_112_Mask);
        results[i] = r;
    }
    match("avx2shuffle");

    // mov32
    for (int i=0; i<8; ++i) {
        r = mov32(val, vidx[i]);
        results[i] = r;
    }
    match("mov32");

    // full scalar
    for (int i=0; i<8; ++i) {
        r = full_scalar(val, vidx[i]);
        results[i] = r;
    }
    match("full_scalar");
    
    printf("\n");
}


static NOINLINE void
test_speed(const unsigned char idx[256], unsigned char val[256])
{
    uint64_t t0, t1;
    int nloop = 1024*1024*8;

    const __m256i* vidx = (const __m256i*)idx;
    __m256i tmp = _mm256_setzero_si256();
    __m256i r;

#ifdef _MSC_VER
    // naive method
    t0 = __rdtsc();
    for (int i=0; i<nloop; i++) {
        for (int j=0; j<8; j++) {
            r = ymm_u8lookup_naive(val, vidx[j]);
            tmp = _mm256_or_si256(tmp, r);
        }
    }
    t1 = __rdtsc();
    printf("naive %f\n", (t1-t0) / (double)nloop);
#endif

    // avx2gather
    __m256i mask_FF = _mm256_set1_epi32(0xFF);
    t0 = __rdtsc();
    for (int i=0; i<nloop; i++) {
        for (int j=0; j<8; j++) {
            r = ymm_u8lookup_avx2gather(val, vidx[j], mask_FF);
            tmp = _mm256_or_si256(tmp, r);
        }
    }
    t1 = __rdtsc();
    printf("avx2gather %f\n", (t1-t0) / (double)nloop);

    // avx2shuffle
    t0 = __rdtsc();
    __m256i m256i_u8_all_16 = _mm256_set1_epi8(0x10);
    __m256i m256i_u8_112_Mask = _mm256_set1_epi8(112);
    __m256i lut[32];
    for (size_t i = 0; i < 16; ++i) {
        __m128i tmp = _mm_loadu_si128((__m128i*)(val + i * 16));
        lut[i] = _mm256_broadcastsi128_si256(tmp);
    }
    for (int i=0; i<nloop; i++) {
        for (int j=0; j<8; j++) {
            r = ymm_u8lookup_avx2shuffle<16>(lut, vidx[j], m256i_u8_all_16, m256i_u8_112_Mask);
            tmp = _mm256_or_si256(tmp, r);
        }
    }
    t1 = __rdtsc();
    printf("avx2shuffle %f\n", (t1-t0) / (double)nloop);

    // mov32
    t0 = __rdtsc();
    for (int i=0; i<nloop; i++) {
        for (int j=0; j<8; j++) {
            r = mov32(val, vidx[j]);
            tmp = _mm256_or_si256(tmp, r);
        }
    }
    t1 = __rdtsc();
    printf("mov32 %f\n", (t1-t0) / (double)nloop);

    // full scalar
    t0 = __rdtsc();
    for (int i=0; i<nloop; i++) {
        for (int j=0; j<8; j++) {
            r = full_scalar(val, vidx[j]);
            tmp = _mm256_or_si256(tmp, r);
        }
    }
    t1 = __rdtsc();
    printf("full scalar %f\n", (t1-t0) / (double)nloop);

#ifdef _MSC_VER
    int64_t total = tmp.m256i_i64[0] + tmp.m256i_i64[1] + tmp.m256i_i64[2] + tmp.m256i_i64[3];
#else
    int64_t total = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
    printf("%lld\n\n", (total+1LL)/total);
}

int
main(int argc, char **argv)
{
    unsigned char mem_idx[256];
    unsigned char mem_val[256];

    int seed = 0;
    int size = 4096;

    if (argc > 1) {
        srand(atoi(argv[1]));
    } else {
        srand(0);
    }

    for (int i=0; i<256; i++) {
        mem_val[i] = i;
    }

    for (int i=0; i<256; i++) {
        mem_idx[i] = 255 - i;
    }

    test_results(mem_idx, mem_val);

    test_speed(mem_idx, mem_val);
    test_speed(mem_idx, mem_val);

#ifdef _MSC_VER
    _getch();
#endif
    return 0;
}

