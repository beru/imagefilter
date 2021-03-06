
#include "common.h"
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
	threads.SetUp((unsigned char)nThreads);
	const size_t partSize = size / nThreads;
	const size_t partHeight = height / nThreads;
	
	blur_1b::Parameter pCommon;
	pCommon.width = (uint16_t)width;
	pCommon.height = (uint16_t)partHeight;
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
			p.height = (uint16_t)(height - partHeight * i);
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

	printf("%zd %zd %zd %p\n", width, height, lineSize, pDest);

	_getch();
	return 0;
}

