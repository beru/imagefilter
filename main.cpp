#include <assert.h>

#include <vector>
#include <algorithm>
#include <cmath>

#include "ReadImage/ReadImage.h"
#include "ReadImage/File.h"

#include "blur_1b.h"
#include "timer.h"
#include "ThreadPool.h"

#include <conio.h>
#include "sym.h"

#include <immintrin.h>

void blur_test(
    size_t nThreads,
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
	Threads<blur_1b::Parameter> threads;
	threads.SetUp(nThreads);
	const size_t partSize = size / nThreads;
	const size_t partHeight = height / nThreads;
	
	blur_1b::Parameter pCommon;
	pCommon.width = width;
	pCommon.height = partHeight;
	pCommon.srcLineOffsetBytes =
	pCommon.workLineOffsetBytes =
	pCommon.destLineOffsetBytes = lineSize;
	pCommon.radius = 10;
	pCommon.iterationCount = 1;
	std::vector<blur_1b::Parameter> params(nThreads);
	for (size_t i=0; i<nThreads; ++i) {
		blur_1b::Parameter& p = params[i];
		p = pCommon;
		if (i == nThreads - 1) {
			p.height = height - partHeight * i;
		}
		p.bTop = (i == 0);
		p.bBottom = (i == nThreads-1);
		p.pSrc = pSrc + i * partSize;
		p.pWork = pWork + i * partSize * 2;
		p.pWork2 = pWork2 + i * partSize * 2;
		p.pDest = pDest + i * partSize;
		p.pTotal = _mm_malloc(lineSize * sizeof(int32_t), 64);
		p.pModi = _mm_malloc(lineSize * sizeof(int32_t), 64);
	}
	typedef void (*BlurFuncPtr)(const blur_1b::Parameter& p);
	BlurFuncPtr ptrs[] = {
		blur_1b::test_1,
		blur_1b::test_2,
		blur_1b::test_3,
		blur_1b::test_4,
		blur_1b::test_5_h,
		blur_1b::test_5_v,
		blur_1b::test_5_h,
		blur_1b::test_6_v,
		blur_1b::test_7_h,
		blur_1b::test_7_v,
		blur_1b::test_8,
		blur_1b::test_9,
		blur_1b::test_10,
		blur_1b::test_11,
		blur_1b::test_12,
		blur_1b::test_13,
		blur_1b::test_14,
		blur_1b::test_15,

		blur_1b::test_20,
		blur_1b::test_21,

		blur_1b::memory_copy1,
		blur_1b::memory_copy2,
		blur_1b::memory_copy3,
	};
	
	Timer t;
	Sym sym;

	for (size_t i=0; i<countof(ptrs); ++i) {
		t.Start();
		
		for (size_t j=0; j<1; ++j) {
			threads.Start(ptrs[i], &params[0]);
			threads.Join();
		}
		double sec = t.ElapsedSecond();
		
		std::string name = sym.GetName(ptrs[i]);
		
		printf("%s %f\n", name.c_str(), sec * 1000.0);
	}

}

#define _MM_ALIGN32 __declspec(align(32))
//#define _MM_ALIGN32 alignas(32) 

static _MM_ALIGN32 const __m256i m256i_lo4BitsMask = _mm256_set1_epi8(0x0F);
static _MM_ALIGN32 const __m256i m256i_hi4BitsMask = _mm256_set1_epi8(0xF0);
static _MM_ALIGN32 const __m256i m256i_u8_16_Mask = _mm256_set1_epi8(0x10);
static _MM_ALIGN32 const __m256i m256i_u8_128_Mask = _mm256_set1_epi8(0x80);
static _MM_ALIGN32 const __m256i m256i_u8_112_Mask = _mm256_set1_epi8(112);

void setTable(uint8_t table[256], double gamma) {
    for (size_t i=0; i<256; ++i) {
        double di = (double)i / 255.0;
        table[i] = std::pow(di, gamma) * 255.0 + 0.5;
    }
}

void setLUT(__m256i lut[16], const uint8_t table[256]) {
    __m128i val;
	for (size_t i=0; i<16; ++i) {
		val = _mm_loadu_si128((__m128i*)(table + i * 16));
		lut[i] = _mm256_broadcastsi128_si256(val);
	}
}
__forceinline __m256i mm256_u8gather_epu8(const __m256i lut[16], __m256i vindex) {

#if 0

	__m256i ret;
	const uint8_t* lut2 = (const uint8_t*)lut;

#if 1
	for (int i=0; i<32; ++i) {
		ret.m256i_u8[i] = lut2[s.m256i_u8[i]];
	}
	return ret;
#else
	extern int   _mm256_extract_epi8 (__m256i /* src */, const int /* index */);
	// crazily slow...
#define SET(idx) ret.m256i_u8[idx] = lut2[_mm256_extract_epi8(s, idx)]
	SET(0);
	SET(1);
	SET(2);
	SET(3);
	SET(4);
	SET(5);
	SET(6);
	SET(7);
	SET(8);
	SET(9);
	SET(10);
	SET(11);
	SET(12);
	SET(13);
	SET(14);
	SET(15);
	SET(16);
	SET(17);
	SET(18);
	SET(19);
	SET(20);
	SET(21);
	SET(22);
	SET(23);
	SET(24);
	SET(25);
	SET(26);
	SET(27);
	SET(28);
	SET(29);
	SET(30);
	SET(31);
	return ret;
#endif

#else

	__m256i s = vindex;
    __m256i tmp;
    __m256i ret;

    tmp = _mm256_adds_epu8(s, m256i_u8_112_Mask);
    s = _mm256_sub_epi8(s, m256i_u8_16_Mask);
    ret = _mm256_shuffle_epi8(_mm256_load_si256(lut), tmp);

#define LOOKUP(idx) \
    tmp = _mm256_adds_epu8(s, m256i_u8_112_Mask);\
    s = _mm256_sub_epi8(s, m256i_u8_16_Mask);\
    tmp = _mm256_shuffle_epi8(_mm256_load_si256(lut+idx), tmp);\
    ret = _mm256_adds_epu8(ret, tmp);

	// a heck a lot of instructions needed...
//    LOOKUP(0)
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
    return ret;

#endif
}

void gamma_correction_test(
    size_t nThreads,
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
    _MM_ALIGN32 uint8_t table0[256];
    _MM_ALIGN32 uint8_t table1[256];
	setTable(table0, 2.2/1.0);
	setTable(table1, 1.0/2.2);

	for (int z=0; z<100; ++z) {

#if 1
	__m256i lut0[16];
	__m256i lut1[16];
	setLUT(lut0, table0);
	setLUT(lut1, table1);
    const uint8_t* pSrcLine = pSrc;
    uint8_t* pDstLine = pDest;
    __m256i byteMask = _mm256_set1_epi16(0x00FF);
    __m256i zero = _mm256_setzero_si256();
    const size_t xCnt = (width + 31) / 32;
    for (size_t y=0; y<height/2; ++y) {
        const __m256i* pSrcLine1 = (const __m256i*) pSrcLine;
        const __m256i* pSrcLine2 = (const __m256i*) (pSrcLine + lineSize);
        __m128i* pDstLine1 = (__m128i*) pDstLine;
        for (size_t x=0; x<xCnt; ++x) {
            __m256i s1 = _mm256_loadu_si256(pSrcLine1+x);
            __m256i s2 = _mm256_loadu_si256(pSrcLine2+x);
            s1 = mm256_u8gather_epu8(lut0, s1);
            s2 = mm256_u8gather_epu8(lut0, s2);
#if 1
            __m256i sn = _mm256_avg_epu8(s1, s2);
            __m256i sn2 = _mm256_srli_si256(sn, 1);
            sn = _mm256_and_si256(sn, byteMask);
            sn2 = _mm256_and_si256(sn2, byteMask);
            sn = _mm256_add_epi16(sn, sn2);
            sn = _mm256_srli_epi16(sn, 1);
            sn = _mm256_packus_epi16(sn, zero);
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
            s1_0 = _mm256_packus_epi16(s1_0, zero);
			s1_0 = _mm256_permute4x64_epi64(s1_0, _MM_SHUFFLE(0, 0, 2, 0));
			__m256i sn = s1_0;
#endif
            sn = mm256_u8gather_epu8(lut1, sn);
            _mm_storeu_si128(pDstLine1+x, _mm256_castsi256_si128(sn));
        }
        pSrcLine += lineSize * 2;
        pDstLine += lineSize;
    }
#else
    const uint8_t* pSrcLine = pSrc;
    uint8_t* pDstLine = pDest;
    for (size_t y=0; y<height/2; ++y) {
        const uint8_t* pSrcLine2 = pSrcLine + lineSize;
        for (size_t x=0; x<width/2; ++x) {
            uint8_t s0 = pSrcLine[x*2];
            uint8_t s1 = pSrcLine[x*2+1];
            uint8_t s2 = pSrcLine2[x*2];
            uint8_t s3 = pSrcLine2[x*2+1];
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
	ReadImageData(fo, pSrc, lineSize, palettes);
	fclose(f);
	
	for (size_t i=0; i<size; ++i) {
		pSrc[i] = palettes[4 * pSrc[i]];
	}
	
	SYSTEM_INFO si;
	GetSystemInfo(&si);
	
#ifdef _DEBUG
	const size_t nThreads = 1;
//	const size_t nThreads = 2;
//	const size_t nThreads = 4;
#else
	const size_t nThreads = si.dwNumberOfProcessors;
//	const size_t nThreads = 1;
#endif

#if 0
    blur_test(
        nThreads,
        width,
        height,
        lineSize,
        size,
        pSrc,
        pWork,
        pWork2,
        pDest
    );
#endif

    gamma_correction_test(
        nThreads,
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

