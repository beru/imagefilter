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

static const __m256i m256i_lo4BitsMask = _mm256_set1_epi8(0x0F);
static const __m256i m256i_hi4BitsMask = _mm256_set1_epi8(0xF0);
static const __m256i m256i_u8_16_Mask = _mm256_set1_epi8(0x10);
static const __m256i m256i_u8_128_Mask = _mm256_set1_epi8(0x80);

struct GammaCorrector
{
    double gamma;
    uint8_t table[256];
    __m256i lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7, lut8, lut9,
            lutA, lutB, lutC, lutD, lutE, lutF;

    GammaCorrector(double gamma)
    {
        Init(gamma);
    }

    void Init(double gamma) {
        this->gamma = gamma;
        for (size_t i=0; i<256; ++i) {
            double di = (double)i / 255.0;
            table[i] = std::pow(di, gamma) * 255.0 + 0.5;
        }
        __m128i val;
        val = _mm_loadu_si128((__m128i*)(table + 0 * 16));
        lut0 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 1 * 16));
        lut1 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 2 * 16));
        lut2 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 3 * 16));
        lut3 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 4 * 16));
        lut4 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 5 * 16));
        lut5 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 6 * 16));
        lut6 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 7 * 16));
        lut7 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 8 * 16));
        lut8 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 9 * 16));
        lut9 = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 0xA * 16));
        lutA = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 0xB * 16));
        lutB = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 0xC * 16));
        lutC = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 0xD * 16));
        lutD = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 0xE * 16));
        lutE = _mm256_broadcastsi128_si256(val);
        val = _mm_loadu_si128((__m128i*)(table + 0xF * 16));
        lutF = _mm256_broadcastsi128_si256(val);
    }

    __m256i conv(__m256i s) {
        //return s;

        __m256i maskLo = m256i_lo4BitsMask;
        __m256i maskHi = m256i_hi4BitsMask;
        __m256i mask16 = m256i_u8_16_Mask;
        __m256i mask128 = m256i_u8_128_Mask;
        __m256i zero = _mm256_setzero_si256();
        __m256i wrk = s;
        __m256i tmp;
        __m256i lt128, gt15;
        __m256i ret;

        tmp = _mm256_and_si256(wrk, m256i_u8_128_Mask);
        lt128 = _mm256_cmpeq_epi8(tmp, zero);
        wrk = _mm256_blendv_epi8(zero, s, lt128);
        ret = zero;
        //ret = _mm256_shuffle_epi8(lut0, wrk);

#define LOOKUP(name) \
        gt15 = _mm256_cmpgt_epi8(wrk, maskLo);\
        tmp = _mm256_shuffle_epi8(lut ## name, wrk);\
        wrk = _mm256_subs_epu8(wrk, mask16);\
        ret = _mm256_blendv_epi8(tmp, ret, gt15);

        LOOKUP(0)
        LOOKUP(1)
        LOOKUP(2)
        LOOKUP(3)
        //LOOKUP(4)
        //LOOKUP(5)
        //LOOKUP(6)
        //LOOKUP(7)
        //LOOKUP(8)
        //LOOKUP(9)
        //LOOKUP(A)
        //LOOKUP(B)
        //LOOKUP(C)
        //LOOKUP(D)
        //LOOKUP(E)
        //LOOKUP(F)
        return ret;
    }

};

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

#if 1
    GammaCorrector gammaCorrector0(2.2/1.0);
    GammaCorrector gammaCorrector1(1.0/2.2);
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
            s1 = gammaCorrector0.conv(s1);
            s2 = gammaCorrector0.conv(s2);
            __m256i sn = _mm256_avg_epu8(s1, s2);
            __m256i sn2 = _mm256_srli_si256(sn, 1);
            sn = _mm256_and_si256(sn, byteMask);
            sn2 = _mm256_and_si256(sn2, byteMask);
            sn = _mm256_add_epi16(sn, sn2);
            sn = _mm256_srli_epi16(sn, 1);
            sn = _mm256_packs_epi16(sn, zero);
            sn = _mm256_permute4x64_epi64(sn, _MM_SHUFFLE(0, 0, 2, 0));
            sn = gammaCorrector1.conv(sn);
            _mm_storeu_si128(pDstLine1+x, _mm256_castsi256_si128(sn));
        }
        pSrcLine += lineSize * 2;
        pDstLine += lineSize;
    }
#else
	double gamma = 2.2;
    uint8_t table1[256];
    uint8_t table0[256];
    for (size_t i=0; i<256; ++i) {
        double di = (double)i / 255.0;
        table1[i] = std::pow(di, gamma) * 255.0 + 0.5;
        table0[i] = std::pow(di, 1.0/gamma) * 255.0 + 0.5;
    }
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

