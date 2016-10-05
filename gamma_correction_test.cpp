
#include <assert.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <conio.h>
#include <immintrin.h>

#include "common.h"
#include "ReadImage/ReadImage.h"
#include "ReadImage/File.h"
#include "timer.h"
#include "sym.h"

namespace {

#define _MM_ALIGN32 __declspec(align(32))
template <typename T>
__forceinline void setTable(T table[256], double gamma) {
	for (size_t i = 0; i < 256; ++i) {
		double di = (double)i / 255.0;
		table[i] = (T)(std::pow(di, gamma) * 255.0 + 0.5);
	}
}

__forceinline void setLUT(__m256i lut[16], const uint8_t table[256]) {
	__m128i val;
	for (size_t i = 0; i < 16; ++i) {
		val = _mm_loadu_si128((__m128i*)(table + i * 16));
		lut[i] = _mm256_broadcastsi128_si256(val);
	}
}

//#define USE_GATHER

#ifdef USE_GATHER
__forceinline
__m256i mm256_u8gather_epu8(const uint8_t* lut, __m256i vindex, __m256i andMask) {

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

#else

template <unsigned N>
__forceinline
__m256i __vectorcall mm256_u8gather_epu8(const __m256i lut[N], __m256i vindex,
										  __m256i m256i_u8_16_Mask, __m256i m256i_u8_112_Mask) {

	// a heck a lot of instructions needed...
	//LOOKUP(0)
	__m256i tmp = _mm256_adds_epu8(vindex, m256i_u8_112_Mask);
	__m256i s = _mm256_sub_epi8(vindex, m256i_u8_16_Mask);
	__m256i ret = _mm256_shuffle_epi8(lut[0], tmp);

#define LOOKUP(idx) \
	tmp = _mm256_adds_epu8(s, m256i_u8_112_Mask);\
	s = _mm256_sub_epi8(s, m256i_u8_16_Mask);\
	tmp = _mm256_shuffle_epi8(lut[idx], tmp);\
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

}

#endif

void gamma_correction_test(
	size_t width,
	size_t height,
	size_t lineSize,
	size_t size,
	const uint8_t* pSrc,
	uint8_t* pWork,
	uint8_t* pWork2,
	uint8_t* pDest
)
{
	Timer t;
	t.Start();

	double gamma = 2.2;
	uint8_t table0[256];
	uint8_t table1[256];
	setTable(table0, 2.2 / 1.0);
	setTable(table1, 1.0 / 2.2);

	__m256i byteMask = _mm256_set1_epi16(0x00FF);
#ifdef USE_GATHER
	__m256i andMask = _mm256_set1_epi32(0xFF);
#else
	__m256i lut0[16];
	__m256i lut1[16];
	setLUT(lut0, table0);
	setLUT(lut1, table1);
	__m256i m256i_u8_16_Mask = _mm256_set1_epi8(0x10);
	__m256i m256i_u8_112_Mask = _mm256_set1_epi8(112);
#endif

	for (int z = 0; z < 1000; ++z) {

#if 1
		const uint8_t* pSrcLine = pSrc;
		uint8_t* pDstLine = pDest;
		const size_t xCnt = (width + 31) / 32;
		for (size_t y = 0; y < height / 2; ++y) {
			const __m256i* pSrcLine1 = (const __m256i*) pSrcLine;
			const __m256i* pSrcLine2 = (const __m256i*) (pSrcLine + lineSize);
			__m128i* pDstLine1 = (__m128i*) pDstLine;
			for (size_t x = 0; x < xCnt; ++x) {
				__m256i s1 = _mm256_loadu_si256(pSrcLine1 + x);
				__m256i s2 = _mm256_loadu_si256(pSrcLine2 + x);
#ifdef USE_GATHER
				s1 = mm256_u8gather_epu8(table0, s1, andMask);
				s2 = mm256_u8gather_epu8(table0, s2, andMask);
#else
				s1 = mm256_u8gather_epu8<16>(lut0, s1, m256i_u8_16_Mask, m256i_u8_112_Mask);
				s2 = mm256_u8gather_epu8<16>(lut0, s2, m256i_u8_16_Mask, m256i_u8_112_Mask);
#endif
				
#if 1
				__m256i sn = _mm256_avg_epu8(s1, s2);
				__m256i sn2 = _mm256_srli_si256(sn, 1);
				sn = _mm256_and_si256(sn, byteMask);
				sn2 = _mm256_and_si256(sn2, byteMask);
				sn = _mm256_add_epi16(sn, sn2);
				sn = _mm256_srli_epi16(sn, 1);
				sn = _mm256_packus_epi16(sn, _mm256_setzero_si256());
				sn = _mm256_permute4x64_epi64(sn, _MM_SHUFFLE(0, 0, 2, 0));
#else
				__m256i s1_0 = _mm256_and_si256(s1, byteMask);
				__m256i s1_1 = _mm256_and_si256(_mm256_srli_si256(s1, 1), byteMask);
				__m256i s2_0 = _mm256_and_si256(s2, byteMask);
				__m256i s2_1 = _mm256_and_si256(_mm256_srli_si256(s2, 1), byteMask);
				s1_0 = _mm256_add_epi16(s1_0, s1_1);
				s2_0 = _mm256_add_epi16(s2_0, s2_1);
				s1_0 = _mm256_add_epi16(s1_0, s2_0);
				s1_0 = _mm256_srli_epi16(s1_0, 2);
				s1_0 = _mm256_packus_epi16(s1_0, _mm256_setzero_si256());
				s1_0 = _mm256_permute4x64_epi64(s1_0, _MM_SHUFFLE(0, 0, 2, 0));
				__m256i sn = s1_0;
#endif

#ifdef USE_GATHER
				sn = mm256_u8gather_epu8(table1, sn, andMask);
#else
				sn = mm256_u8gather_epu8<16>(lut1, sn, m256i_u8_16_Mask, m256i_u8_112_Mask);
#endif
				
				_mm_storeu_si128(pDstLine1 + x, _mm256_castsi256_si128(sn));
			}
			pSrcLine += lineSize * 2;
			pDstLine += lineSize;
		}
#else
		const uint8_t* pSrcLine = pSrc;
		uint8_t* pDstLine = pDest;
		for (size_t y = 0; y < height / 2; ++y) {
			const uint8_t* pSrcLine2 = pSrcLine + lineSize;
			for (size_t x = 0; x < width / 2; ++x) {
				uint8_t s0 = pSrcLine[x * 2];
				uint8_t s1 = pSrcLine[x * 2 + 1];
				uint8_t s2 = pSrcLine2[x * 2];
				uint8_t s3 = pSrcLine2[x * 2 + 1];
#if 1
				int sum = table1[s0] + table1[s1] + table1[s2] + table1[s3];
				pDstLine[x] = table0[sum >> 2];
#else
				int sum = s0 + s1 + s2 + s3;
				pDstLine[x] = sum >> 2;
#endif
			}
			pSrcLine += lineSize * 2;
			pDstLine += lineSize;
		}
#endif


	}

	double sec = t.ElapsedSecond();
	printf("%f\n", sec * 1000.0);
}

} // namespace {

int main(int argc, char* argv[])
{
	if (argc < 2) {
		printf("specify filename\n");
		return 1;
	}
	
	FILE* f = fopen(argv[1], "rb");
	if (!f) {
		printf("failed to open file : %s\n", argv[1]);
		return 1;
	}
	File fo(f);
	ImageInfo imageInfo;
	ReadImageInfo(fo, imageInfo);
	
	size_t width = imageInfo.width;
	size_t height = imageInfo.height;
	assert(imageInfo.bitsPerSample == 8 && imageInfo.samplesPerPixel == 1);

	size_t lineSize = (width + 63) & (~63);
	const size_t size = lineSize * height;
	uint8_t* pSrc = (uint8_t*) _mm_malloc(size, 64);
	uint8_t* pDest = (uint8_t*) _mm_malloc(size, 64);
	uint8_t* pWork = (uint8_t*) _mm_malloc(size*4, 64);
	uint8_t* pWork2 = (uint8_t*) _mm_malloc(size*4, 64);
	
	uint8_t palettes[256 * 4];
	ReadImageData(fo, pSrc, (int)lineSize, palettes);
	fclose(f);
	
	for (size_t i=0; i<size; ++i) {
		pSrc[i] = palettes[4 * pSrc[i]];
	}
	
	gamma_correction_test(
		width,
		height,
		lineSize,
		size,
		pSrc,
		pWork,
		pWork2,
		pDest
	);

	printf("%zd %zd %zd %p\n", width, height, lineSize, pDest);

	_getch();
	return 0;
}

