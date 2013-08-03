//
// Copyright (c) 2010 Katsuhisa Yuasa <berupon [at] gmail.com>
//
// ------
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "blur_1b.h"
#include <algorithm>
#include <emmintrin.h>
//#include <smmintrin.h>
#if _MSC_VER >= 1700
#include "immintrin.h"
#endif
#include "RingLinePtr.h"

namespace blur_1b {

static const int SHIFT = 16;	// do not change this value

#define BLUR_EXTRACT_PARAMS \
	const uint8_t* const pSrc = p.pSrc;\
	const uint16_t width = p.width;\
	const uint16_t height = p.height;\
	bool bTop = p.bTop;\
	bool bBottom = p.bBottom;\
	const ptrdiff_t srcLineOffsetBytes = p.srcLineOffsetBytes;\
	uint8_t* pDest = p.pDest;\
	const ptrdiff_t destLineOffsetBytes = p.destLineOffsetBytes;\
	uint8_t* pWork = p.pWork;\
	uint8_t* pWork2 = p.pWork2;\
	const ptrdiff_t workLineOffsetBytes = p.workLineOffsetBytes;\
	void* pTotal = p.pTotal;\
	const uint8_t radius = p.radius;\
	const uint8_t iterationCount = p.iterationCount;

template <typename T>
class Image {
private:
	T* mPtr;
	size_t mWidth;
	size_t mHeight;
	ptrdiff_t mLineOffsetBytes;
public:
	Image(T* ptr, size_t width, size_t height, ptrdiff_t lineOffsetBytes)
		:
		mPtr(ptr),
		mWidth(width),
		mHeight(height),
		mLineOffsetBytes(lineOffsetBytes)
	{
	}
	
	Image() {
	}
	
	Image(const Image& i)
		:
		mPtr(i.mPtr),
		mWidth(i.mWidth),
		mHeight(i.mHeight),
		mLineOffsetBytes(i.mLineOffsetBytes)
	{
	}
	
	Image& operator = (const Image& i) {
		mPtr = i.mPtr;
		mWidth = i.mWidth;
		mHeight = i.mHeight;
		mLineOffsetBytes = i.mLineOffsetBytes;
		return *this;
	}
	
	T get(size_t x, size_t y) const {
		if (x >= mWidth || y >= mHeight) {
			return T();
		}
		T* pLine = mPtr;
		OffsetPtr(pLine, mLineOffsetBytes * y);
		return pLine[x];
	}
	
	void set(size_t x, size_t y, T val) {
		if (x >= mWidth || y >= mHeight) {
			return;
		}
		T* pLine = mPtr;
		OffsetPtr(pLine, mLineOffsetBytes * y);
		pLine[x] = val;
	}
};

void memory_copy1(const Parameter& p) {
	memcpy((void*)p.pSrc, p.pDest, p.srcLineOffsetBytes * p.height);
}

void memory_copy2(const Parameter& p) {
	const size_t cnt = (p.srcLineOffsetBytes * p.height) / 64;
	const __m128i* src = (const __m128i*) p.pSrc;
	__m128i* dst = (__m128i*) p.pDest;
	for (size_t i=0; i<cnt; ++i) {
		__m128i src0 = src[0];
		__m128i src1 = src[1];
		__m128i src2 = src[2];
		__m128i src3 = src[3];
		_mm_stream_si128(dst+0, src0);
		_mm_stream_si128(dst+1, src1);
		_mm_stream_si128(dst+2, src2);
		_mm_stream_si128(dst+3, src3);
		src += 4;
		dst += 4;
	}
}

void memory_copy3(const Parameter& p) {
	const size_t cnt = (p.srcLineOffsetBytes * p.height) / 64;
	const __m256i* src = (const __m256i*) p.pSrc;
	__m256i* dst = (__m256i*) p.pDest;
	for (size_t i=0; i<cnt; ++i) {
#if 0
		_mm256_stream_si256(dst+0, src[0]);
		_mm256_stream_si256(dst+1, src[1]);
#else
		dst[0] = src[0];
		dst[1] = src[1];
#endif
		src += 2;
		dst += 2;
	}
}

void test_1(const Parameter& p) {

	BLUR_EXTRACT_PARAMS;
	
	const int r = radius;
	double len = r * 2 + 1;
	double area = len * len;
	double invArea = 1.0 / area;
	
	Image<uint8_t> src((uint8_t*)pSrc, width, height, srcLineOffsetBytes);
	Image<uint8_t> dest(pDest, width, height, destLineOffsetBytes);
	Image<uint8_t> work(pWork, width, height, workLineOffsetBytes);
	Image<uint8_t> work2(pWork2, width, height, workLineOffsetBytes);
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		Image<uint8_t> from = (n == 0) ? src : ((n % 2 == 1) ? work : work2);
		Image<uint8_t> to = (n == iterationCount - 1) ? dest : ((n % 2 == 0) ? work : work2);
		for (size_t y=0; y<height; ++y) {
			for (size_t x=0; x<width; ++x) {
				unsigned int total = 0;
				for (int ky=-r; ky<=r; ++ky) {
					for (int kx=-r; kx<=r; ++kx) {
						total += from.get(x+kx, y+ky);
					}
				}
				to.set(x, y, (uint8_t)(total * invArea + 0.5));
			}
		}
		
	}
}

void test_2(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int r = p.radius;
	double len = r * 2 + 1;
	double area = len * len;
	double invArea = 1.0 / area;
	
	for (size_t n=0; n<iterationCount; ++n) {
	
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			pFrom = ((n % 2 == 1) ? pWork : pWork2);
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = ((n % 2 == 0) ? pWork : pWork2);
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		for (size_t y=0; y<height; ++y) {
			for (size_t x=0; x<width; ++x) {
				unsigned int total = 0;
				int kys = y - r;
				int kye = y + r;
				if (bTop) {
					kys = std::max<int>(0, kys);
				}
				if (bBottom) {
					kye = std::min<int>(height, kye);
				}
				const int kxs = std::max<int>(0, x-r);
				const int kxe = std::min<int>(width, x+r);
				const uint8_t* pFromLine = pFrom;
				OffsetPtr(pFromLine, kys*fromLineOffsetBytes);
				for (int ky=kys; ky<kye; ++ky) {
					for (int kx=kxs; kx<=kxe; ++kx) {
						total += pFromLine[kx];
					}
					OffsetPtr(pFromLine, fromLineOffsetBytes);
				}
				pTo[x] = (uint8_t)(total * invArea + 0.5);
			}
			OffsetPtr(pTo, toLineOffsetBytes);
		}
	
	}
}

void test_3(const Parameter& p) {

	BLUR_EXTRACT_PARAMS;
	
	int r = radius;
	double len = r * 2 + 1;
	double area = len * len;
	double invLen = 1.0 / len;
	double invArea = 1.0 / area;

	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			pFrom = pWork2;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		// horizontal
		{
			uint8_t* pWorkLine = pWork;
			const uint8_t* pFromLine = pFrom;
			for (size_t y=0; y<height; ++y) {
				for (size_t x=0; x<width; ++x) {
					unsigned int total = 0;
					const int kxs = std::max<int>(0, x-r);
					const int kxe = std::min<int>(width, x+r);
					for (int kx=kxs; kx<=kxe; ++kx) {
						total += pFromLine[kx];
					}
					pWorkLine[x] = (uint8_t)(total * invLen + 0.5);
				}
				OffsetPtr(pWorkLine, workLineOffsetBytes);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
			}
		}
		// vertical
		{
			const uint8_t* pWorkLine = pWork;
			uint8_t* pToLine = pTo;
			for (size_t x=0; x<width; ++x) {
				pToLine = pTo;
				pToLine += x;
				for (size_t y=0; y<height; ++y) {
					unsigned int total = 0;
					int kys = y - r;
					int kye = y + r;
					if (bTop) {
						kys = std::max<int>(0, kys);
					}
					if (bBottom) {
						kye = std::min<int>(height, kye);
					}
					pWorkLine = pWork;
					OffsetPtr(pWorkLine, kys*workLineOffsetBytes);
					for (int ky=kys; ky<=kye; ++ky) {
						total += pWorkLine[x];
						OffsetPtr(pWorkLine, workLineOffsetBytes);
					}
					*pToLine = (uint8_t)(total * invLen + 0.5);
					OffsetPtr(pToLine, toLineOffsetBytes);
				}
			}
		}
		
	}
}

void test_4(const Parameter& p) {

	BLUR_EXTRACT_PARAMS;
	
	int r = radius;
	int len = r * 2 + 1;
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			pFrom = pWork2;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		// horizontal
		{
			uint8_t* pWorkLine = pWork;
			const uint8_t* pFromLine = pFrom;
			for (size_t y=0; y<height; ++y) {
				for (size_t x=0; x<width; ++x) {
					unsigned int total = 0;
					const int kxs = std::max<int>(0, x-r);
					const int kxe = std::min<int>(width, x+r);
					for (int kx=kxs; kx<=kxe; ++kx) {
						total += pFromLine[kx];
					}
					pWorkLine[x] = (total * invLen) >> SHIFT;
				}
				OffsetPtr(pWorkLine, workLineOffsetBytes);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
			}
		}
		// vertical
		{
			const uint8_t* pWorkLine = pWork;
			uint8_t* pToLine = pTo;
			for (size_t x=0; x<width; ++x) {
				pToLine = pTo + x;
				for (size_t y=0; y<height; ++y) {
					unsigned int total = 0;
					int kys = y - r;
					int kye = y + r;
					if (bTop) {
						kys = std::max<int>(0, kys);
					}
					if (bBottom) {
						kye = std::min<int>(height, kye);
					}
					pWorkLine = pWork;
					OffsetPtr(pWorkLine, kys*workLineOffsetBytes);
					for (int ky=kys; ky<=kye; ++ky) {
						total += pWorkLine[x];
						OffsetPtr(pWorkLine, workLineOffsetBytes);
					}
					*pToLine = (total * invLen) >> SHIFT;
					OffsetPtr(pToLine, toLineOffsetBytes);
				}
			}
		}
	}
}

void test_5_h(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int r = std::min<int>(height, std::min<int>(width, radius));
	int len = r * 2 + 1; // diameter
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	// horizontal
	const size_t kxs0 = std::min<size_t>(width, 1 + r);
	const size_t kxe0 = (size_t) std::max<int>(0, width - r);
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			// TODO: make it compact
			if (iterationCount & 1) {
				pFrom = (n & 1) ? pWork : pWork2;
			}else {
				pFrom = (n & 1) ? pWork2 : pWork;
			}
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pWork;
			toLineOffsetBytes = workLineOffsetBytes;
		}else {
			// TODO: make it compact
			if (iterationCount & 1) {
				pTo = (n & 1) ? pWork2 : pWork;
			}else {
				pTo = (n & 1) ? pWork : pWork2;
			}
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		const uint8_t* pFromLine = pFrom;
		uint8_t* pToLine = pTo;
		
		for (size_t y=0; y<height; ++y) {
			int total = *pFromLine;
			for (size_t kx=1; kx<kxs0; ++kx) {
				total += pFromLine[kx] * 2;
			}
			pToLine[0] = (total * invLen) >> SHIFT;
			for (size_t x=1; x<kxs0; ++x) {
				assert(kxs0 >= x);
				total -= pFromLine[kxs0 - x];
				total += pFromLine[kxs0 + x - 1];
				pToLine[x] = (total * invLen) >> SHIFT;
			}
			for (size_t x=kxs0; x<kxe0; ++x) {
				total -= pFromLine[x - r - 1];
				total += pFromLine[x + r];
				pToLine[x] = (total * invLen) >> SHIFT;
			}
			for (size_t x=kxe0,cnt=0; x<width; ++x, ++cnt) {
				total -= pFromLine[kxe0 - r + cnt];
				total += pFromLine[width - 1 - cnt];
				pToLine[x] = (total * invLen) >> SHIFT;
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
	}
}

void test_5_v(const Parameter& p) {

	BLUR_EXTRACT_PARAMS;
	
	int r = std::min<int>(height, std::min<int>(width, radius));
	int len = r * 2 + 1; // diameter
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	// vertical
	int kys0 = 1;
	int kye0 = height;
	if (bTop) {
		kys0 = std::min<int>(height, 1 + r);
	}
	if (bBottom) {
		kye0 = std::max<int>(0, height - r);
	}
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pWork;
			fromLineOffsetBytes = workLineOffsetBytes;
		}else {
			pFrom = (n & 1) ? pWork2 : pWork;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = (n & 1) ? pWork : pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		const uint8_t* pFromLine = pFrom;
		const uint8_t* pFromLine2 = pFromLine;
		uint8_t* pToLine = pTo;
		
		for (size_t x=0; x<width; ++x) {
			
			pToLine = pTo + x;
			const uint8_t* pFromBase = pFrom + x;
			
			int total = 0;
			
			if (bTop) {
				pFromLine = pFromBase;
				pFromLine2 = pFromLine;
				OffsetPtr(pFromLine2, r * fromLineOffsetBytes);
				
				total = *pFromLine;
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				for (size_t ky=1; ky<=r; ++ky) {
					total += *pFromLine * 2;
					OffsetPtr(pFromLine, fromLineOffsetBytes);
				}
				
				*pToLine = (total * invLen) >> SHIFT;
				OffsetPtr(pToLine, toLineOffsetBytes);
				
				for (size_t y=1; y<=r; ++y) {
					total -= *pFromLine2;
					total += *pFromLine;
					OffsetPtr(pFromLine2, -fromLineOffsetBytes);
					OffsetPtr(pFromLine, fromLineOffsetBytes);
					*pToLine = (total * invLen) >> SHIFT;
					OffsetPtr(pToLine, toLineOffsetBytes);
				}
			}else {
				pFromLine = pFromBase;
				OffsetPtr(pFromLine, -r * fromLineOffsetBytes);
				pFromLine2 = pFromLine;
				for (int ky=-r; ky<=r; ++ky) {
					total += *pFromLine;
					OffsetPtr(pFromLine, fromLineOffsetBytes);
				}
				*pToLine = (total * invLen) >> SHIFT;
				OffsetPtr(pToLine, toLineOffsetBytes);

			}

			for (size_t y=kys0; y<kye0; ++y) {
				total -= *pFromLine2;
				total += *pFromLine;
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				OffsetPtr(pFromLine2, fromLineOffsetBytes);
				*pToLine = (total * invLen) >> SHIFT;
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
			
			if (bBottom) {
				pFromLine2 = pFromBase;
				OffsetPtr(pFromLine2, (kye0 - r) * fromLineOffsetBytes);
				pFromLine = pFromBase;
				OffsetPtr(pFromLine, (height - 1) * fromLineOffsetBytes);
				for (size_t y=kye0,cnt=0; y<height; ++y, ++cnt) {
					total -= *pFromLine2;
					total += *pFromLine;
					OffsetPtr(pFromLine2, fromLineOffsetBytes);
					OffsetPtr(pFromLine, -fromLineOffsetBytes);
					*pToLine = (total * invLen) >> SHIFT;
					OffsetPtr(pToLine, toLineOffsetBytes);
				}
			}
		}
		
	}
	
}

void test_6_v(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int r = std::min<int>(height, std::min<int>(width, radius));
	int len = r * 2 + 1; // diameter
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	// vertical
	int kys0 = 1;
	int kye0 = height;
	if (bTop) {
		kys0 = std::min<int>(height, 1 + r);
	}
	if (bBottom) {
		kye0 = std::max<int>(0, height - r);
	}
	
	uint16_t* pTotalLine = (uint16_t*) pTotal;
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pWork;
			fromLineOffsetBytes = workLineOffsetBytes;
		}else {
			pFrom = (n & 1) ? pWork2 : pWork;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = (n & 1) ? pWork : pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		const uint8_t* pFromLine = pFrom;
		const uint8_t* pFromLine2 = pFromLine;
		uint8_t* pToLine = pTo;
		
		if (bTop) {
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] = pFromLine[x];
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			for (size_t ky=1; ky<=r; ++ky) {
				for (size_t x=0; x<width; ++x) {
					pTotalLine[x] += pFromLine[x] * 2;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
			OffsetPtr(pFromLine2, r * fromLineOffsetBytes);
			
			for (size_t y=1; y<=r; ++y) {
				for (size_t x=0; x<width; ++x) {
					int total = pTotalLine[x] - pFromLine2[x] + pFromLine[x];
					pToLine[x] = (total * invLen) >> SHIFT;
					pTotalLine[x] = total;
				}
				OffsetPtr(pFromLine2, -fromLineOffsetBytes);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}else {
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] = 0;
			}
			OffsetPtr(pFromLine, -r * fromLineOffsetBytes);
			pFromLine2 = pFromLine;
			for (int y=-r; y<=r; ++y) {
				for (size_t x=0; x<width; ++x) {
					pTotalLine[x] += pFromLine[x];
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		for (size_t y=kys0; y<kye0; ++y) {
			for (size_t x=0; x<width; ++x) {
				int total = pTotalLine[x] - pFromLine2[x] + pFromLine[x];
				pToLine[x] = (total * invLen) >> SHIFT;
				pTotalLine[x] = total;
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			OffsetPtr(pFromLine2, fromLineOffsetBytes);
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		if (bBottom) {
			pFromLine2 = pWork;
			OffsetPtr(pFromLine2, (kye0 - r) * fromLineOffsetBytes);
			pFromLine = pWork;
			OffsetPtr(pFromLine, (height - 1) * fromLineOffsetBytes);
			for (size_t y=kye0,cnt=0; y<height; ++y, ++cnt) {
				for (size_t x=0; x<width; ++x) {
					int total = pTotalLine[x] - pFromLine2[x] + pFromLine[x];
					pToLine[x] = (total * invLen) >> SHIFT;
					pTotalLine[x] = total;
				}
				OffsetPtr(pFromLine2, fromLineOffsetBytes);
				OffsetPtr(pFromLine, -fromLineOffsetBytes);
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}
	}
}

void test_7_h(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int32_t r = std::min<int32_t>(height, std::min<int32_t>(width, radius));
	int32_t len = r * 2 + 1; // diameter
	__m128 mInvLen = _mm_set1_ps(1.0 / len);
	int32_t invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	__m128i mInvLeni =  _mm_set1_epi16(invLen);
	
	// horizontal
	const size_t kxs0 = std::min<size_t>(width, 1 + r);
	const size_t kxe0 = (size_t) std::max<int>(0, width - r);
	const size_t kxs0_16end = kxs0 + (16 - (kxs0 % 16));
	const size_t kxe0_16end = kxe0 - (kxe0 % 16);
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			// TODO: make it compact
			if (iterationCount & 1) {
				pFrom = (n & 1) ? pWork : pWork2;
			}else {
				pFrom = (n & 1) ? pWork2 : pWork;
			}
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pWork;
			toLineOffsetBytes = workLineOffsetBytes;
		}else {
			// TODO: make it compact
			if (iterationCount & 1) {
				pTo = (n & 1) ? pWork2 : pWork;
			}else {
				pTo = (n & 1) ? pWork : pWork2;
			}
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		const uint8_t* pFromLine = pFrom;
		uint8_t* pToLine = pTo;
	
		for (size_t y=0; y<height; ++y) {
			int total = *pFromLine;
			for (size_t kx=1; kx<kxs0; ++kx) {
				total += pFromLine[kx] * 2;
			}
			pToLine[0] = (total * invLen) >> SHIFT;
			for (size_t x=1; x<kxs0; ++x) {
				assert(kxs0 >= x);
				total += - pFromLine[kxs0 - x] + pFromLine[kxs0 + x - 1];
				pToLine[x] = (total * invLen) >> SHIFT;
			}
			for (size_t x=kxs0; x<kxs0_16end; ++x) {
				total += - pFromLine[x - r - 1] + pFromLine[x + r];
				pToLine[x] = (total * invLen) >> SHIFT;
			}
			
			__m128i* mPSub = (__m128i*) (pFromLine + kxs0_16end - r - 1);
			__m128i* mPAdd = (__m128i*) (pFromLine + kxs0_16end + r);
			__m128i mNextSub = _mm_loadu_si128(mPSub++); // hoist loading
			__m128i mNextAdd = _mm_loadu_si128(mPAdd++);
			__m128i* mPWork = (__m128i*) (pToLine + kxs0_16end);

	#if 1
			__m128i mTotal = _mm_set1_epi16(total);
			for (size_t x=kxs0_16end; x<kxe0_16end; x+=16) {
				__m128i mSub = mNextSub;
				__m128i mAdd = mNextAdd;
				mNextSub = _mm_loadu_si128(mPSub++);
				mNextAdd = _mm_loadu_si128(mPAdd++);
				
				__m128i mSub0 = _mm_unpacklo_epi8(mSub, _mm_setzero_si128());
				__m128i mSub1 = _mm_unpackhi_epi8(mSub, _mm_setzero_si128());
				__m128i mAdd0 = _mm_unpacklo_epi8(mAdd, _mm_setzero_si128());
				__m128i mAdd1 = _mm_unpackhi_epi8(mAdd, _mm_setzero_si128());
				
				__m128i mDiff0 = _mm_sub_epi16(mAdd0, mSub0);
				__m128i mDiff1 = _mm_sub_epi16(mAdd1, mSub1);
				mDiff0 = _mm_add_epi16(mDiff0, _mm_slli_si128(mDiff0, 2));
				mDiff0 = _mm_add_epi16(mDiff0, _mm_slli_si128(mDiff0, 4));
				mDiff0 = _mm_add_epi16(mDiff0, _mm_slli_si128(mDiff0, 8));
				__m128i jump = _mm_shufflehi_epi16(mDiff0, _MM_SHUFFLE(3,3,3,3));
				jump = _mm_unpackhi_epi64(jump, jump);
				
				mDiff1 = _mm_add_epi16(mDiff1, _mm_slli_si128(mDiff1, 2));
				mDiff1 = _mm_add_epi16(mDiff1, _mm_slli_si128(mDiff1, 4));
				mDiff1 = _mm_add_epi16(mDiff1, _mm_slli_si128(mDiff1, 8));
				mDiff1 = _mm_add_epi16(mDiff1, jump);
				
				__m128i left = _mm_add_epi16(mTotal, mDiff0);
				__m128i right = _mm_add_epi16(mTotal, mDiff1);
				__m128i left2 = _mm_mulhi_epu16(left, mInvLeni);
				__m128i right2 = _mm_mulhi_epu16(right, mInvLeni);
				__m128i result = _mm_packus_epi16(left2, right2);
	//			_mm_stream_si128(mPWork++, result);
				*mPWork++ = result;
				mTotal = _mm_shufflehi_epi16(right, _MM_SHUFFLE(3,3,3,3));
				mTotal = _mm_unpackhi_epi64(mTotal, mTotal);
			}
			total = _mm_extract_epi16(mTotal, 0);				
	#else
			// SSE2 path				
			__m128i mTotal = _mm_set1_epi32(total); //_mm_shuffle_epi32(_mm_cvtsi32_si128(total), 1);
			for (size_t x=kxs0_16end; x<kxe0_16end; x+=16) {
				__m128i mSub = mNextSub;
				__m128i mAdd = mNextAdd;
				mNextSub = _mm_loadu_si128(mPSub++);
				mNextAdd = _mm_loadu_si128(mPAdd++);
				
				__m128i mSub0 = _mm_unpacklo_epi8(mSub, _mm_setzero_si128());
				__m128i mSub1 = _mm_unpackhi_epi8(mSub, _mm_setzero_si128());
				__m128i mAdd0 = _mm_unpacklo_epi8(mAdd, _mm_setzero_si128());
				__m128i mAdd1 = _mm_unpackhi_epi8(mAdd, _mm_setzero_si128());
				
				__m128i mDiff0 = _mm_sub_epi32(_mm_unpacklo_epi16(mAdd0, _mm_setzero_si128()), _mm_unpacklo_epi16(mSub0, _mm_setzero_si128()));
				__m128i mDiff1 = _mm_sub_epi32(_mm_unpackhi_epi16(mAdd0, _mm_setzero_si128()), _mm_unpackhi_epi16(mSub0, _mm_setzero_si128()));
				__m128i mDiff2 = _mm_sub_epi32(_mm_unpacklo_epi16(mAdd1, _mm_setzero_si128()), _mm_unpacklo_epi16(mSub1, _mm_setzero_si128()));
				__m128i mDiff3 = _mm_sub_epi32(_mm_unpackhi_epi16(mAdd1, _mm_setzero_si128()), _mm_unpackhi_epi16(mSub1, _mm_setzero_si128()));
				
				mDiff0 = _mm_add_epi32(mDiff0, _mm_slli_si128(mDiff0, 4));
				mDiff0 = _mm_add_epi32(mDiff0, _mm_slli_si128(mDiff0, 8));
				mDiff1 = _mm_add_epi32(mDiff1, _mm_slli_si128(mDiff1, 4));
				mDiff1 = _mm_add_epi32(mDiff1, _mm_slli_si128(mDiff1, 8));
				mDiff2 = _mm_add_epi32(mDiff2, _mm_slli_si128(mDiff2, 4));
				mDiff2 = _mm_add_epi32(mDiff2, _mm_slli_si128(mDiff2, 8));
				mDiff3 = _mm_add_epi32(mDiff3, _mm_slli_si128(mDiff3, 4));
				mDiff3 = _mm_add_epi32(mDiff3, _mm_slli_si128(mDiff3, 8));

				mTotal = _mm_add_epi32(mTotal, mDiff0);
				__m128 mfTotal = _mm_cvtepi32_ps(mTotal);
				__m128i mDest0 = _mm_cvttps_epi32(_mm_mul_ps(mfTotal, mInvLen));
				
				mTotal = _mm_shuffle_epi32(mTotal, _MM_SHUFFLE(3,3,3,3));
				mTotal = _mm_add_epi32(mTotal, mDiff1);
				mfTotal = _mm_cvtepi32_ps(mTotal);
				__m128i mDest1 = _mm_cvttps_epi32(_mm_mul_ps(mfTotal, mInvLen));

				mTotal = _mm_shuffle_epi32(mTotal, _MM_SHUFFLE(3,3,3,3));
				mTotal = _mm_add_epi32(mTotal, mDiff2);
				mfTotal = _mm_cvtepi32_ps(mTotal);
				__m128i mDest2 = _mm_cvttps_epi32(_mm_mul_ps(mfTotal, mInvLen));

				mTotal = _mm_shuffle_epi32(mTotal, _MM_SHUFFLE(3,3,3,3));
				mTotal = _mm_add_epi32(mTotal, mDiff3);
				mfTotal = _mm_cvtepi32_ps(mTotal);
				mTotal = _mm_shuffle_epi32(mTotal, _MM_SHUFFLE(3,3,3,3));
				__m128i mDest3 = _mm_cvttps_epi32(_mm_mul_ps(mfTotal, mInvLen));

				*mPWork++ =
					_mm_packus_epi16(
						_mm_packs_epi32(mDest0, mDest1),
						_mm_packs_epi32(mDest2, mDest3)
					)
				;
			}
			total = _mm_cvtsi128_si32(mTotal);
	#endif
				
			for (size_t x=kxe0_16end; x<kxe0; ++x) {
				total += - pFromLine[x - r - 1] + pFromLine[x + r];
				pToLine[x] = (total * invLen) >> SHIFT;
			}
			for (size_t x=kxe0,cnt=0; x<width; ++x, ++cnt) {
				total += - pFromLine[kxe0 - r + cnt] + pFromLine[width - 1 - cnt];
				pToLine[x] = (total * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
			OffsetPtr(pFromLine, fromLineOffsetBytes);
		}
	}
}

void test_7_v(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int32_t r = std::min<int32_t>(height, std::min<int32_t>(width, radius));
	int32_t len = r * 2 + 1; // diameter
	__m128 mInvLen = _mm_set1_ps(1.0 / len);
	int32_t invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	__m128i mInvLeni =  _mm_set1_epi16(invLen);
	uint16_t* pTotalLine = (uint16_t*) pTotal;
	
	// vertical
	struct Worker {
		static __forceinline void process(
			const __m128i* pSubLine,
			const __m128i* pAddLine,
			__m128i* pTotalLine,
			__m128i* pDestLine,
			__m128i mInvLen,
			size_t width
		) {
			const size_t loopCount = width >> 4;
			for (size_t i=0; i<loopCount; ++i) {
				__m128i mSub = pSubLine[i];
				__m128i mAdd = pAddLine[i];
				
				__m128i mTotal0 = pTotalLine[i*2+0];
				__m128i mTotal1 = pTotalLine[i*2+1];
				
				__m128i mSub0 = _mm_unpacklo_epi8(mSub, _mm_setzero_si128());
				__m128i mSub1 = _mm_unpackhi_epi8(mSub, _mm_setzero_si128());
				__m128i mAdd0 = _mm_unpacklo_epi8(mAdd, _mm_setzero_si128());
				__m128i mAdd1 = _mm_unpackhi_epi8(mAdd, _mm_setzero_si128());
				
				__m128i mDiff0 = _mm_sub_epi16(mAdd0, mSub0);
				__m128i mDiff1 = _mm_sub_epi16(mAdd1, mSub1);
				
				mTotal0 = _mm_add_epi16(mTotal0, mDiff0);
				mTotal1 = _mm_add_epi16(mTotal1, mDiff1);
				
				pTotalLine[i*2+0] = mTotal0;
				pTotalLine[i*2+1] = mTotal1;
				
				__m128i mDest0 = _mm_mulhi_epu16(mTotal0, mInvLen);
				__m128i mDest1 = _mm_mulhi_epu16(mTotal1, mInvLen);
				
				__m128i mResult = _mm_packus_epi16(mDest0, mDest1);
				
				_mm_stream_si128(pDestLine+i, mResult);
//				pDestLine[i] = mResult;
			}
			const size_t remainLoopCount = width & 0xF;
			uint8_t* pDest = (uint8_t*) (pDestLine + loopCount);
			uint8_t* pSub = (uint8_t*) (pSubLine + loopCount);
			uint8_t* pAdd = (uint8_t*) (pAddLine + loopCount);
			int16_t* pTotal = (int16_t*) (pTotalLine + loopCount * 2);
			int16_t invLen = _mm_extract_epi16(mInvLen, 0);				
			for (size_t i=0; i<remainLoopCount; ++i) {
				int16_t total = pTotal[i];
				total += pAdd[i] - pSub[i];
				pDest[i] = ((total * invLen) >> SHIFT);
				pTotal[i] = total;
			}
		}
	};
	
	int kys0 = 1;
	int kye0 = height;
	if (bTop) {
		kys0 = std::min<int>(height, 1 + r);
	}
	if (bBottom) {
		kye0 = std::max<int>(0, height - r);
	}
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pWork;
			fromLineOffsetBytes = workLineOffsetBytes;
		}else {
			pFrom = (n & 1) ? pWork2 : pWork;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = (n & 1) ? pWork : pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		const uint8_t* pFromLine = pFrom;
		const uint8_t* pFromLine2 = pFromLine;
		uint8_t* pToLine = pTo;

		if (bTop) {
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] = pFromLine[x];
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			for (size_t ky=1; ky<=r; ++ky) {
				for (size_t x=0; x<width; ++x) {
					pTotalLine[x] += pFromLine[x] * 2;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
			OffsetPtr(pFromLine2, r * fromLineOffsetBytes);
			
			for (size_t y=1; y<=r; ++y) {
				Worker::process((const __m128i*)pFromLine2, (const __m128i*)pFromLine, (__m128i*)pTotalLine, (__m128i*)pToLine, mInvLeni, width);
				OffsetPtr(pFromLine2, -fromLineOffsetBytes);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
			
		}else {
			__m128i* pMTotal = (__m128i*)pTotalLine;
			for (size_t x=0; x<width>>4; ++x) {
				*pMTotal++ = _mm_setzero_si128();
				*pMTotal++ = _mm_setzero_si128();
			}
			for (size_t x=width&0xFFF0; x<width; ++x) {
				pTotalLine[x] = 0;
			}
			OffsetPtr(pFromLine, -r * fromLineOffsetBytes);
			pFromLine2 = pFromLine;
			for (int ky=-r; ky<=r; ++ky) {
				const __m128i* pMWork = (const __m128i*)pFromLine;
				pMTotal = (__m128i*)pTotalLine;
				for (size_t x=0; x<width>>4; ++x) {
					__m128i mData = pMWork[x];
					__m128i mLeft = _mm_unpacklo_epi8(mData, _mm_setzero_si128());
					__m128i mRight = _mm_unpackhi_epi8(mData, _mm_setzero_si128());
					
					__m128i totalLeft = *pMTotal;
					__m128i totalRight = *(pMTotal+1);
					*pMTotal++ = _mm_add_epi16(totalLeft, mLeft);
					*pMTotal++ = _mm_add_epi16(totalRight, mRight);
				}
				for (size_t x=width&0xFFF0; x<width; ++x) {
					pTotalLine[x] += pFromLine[x];
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		for (int y=kys0; y<kye0; ++y) {
			Worker::process((const __m128i*)pFromLine2, (const __m128i*)pFromLine, (__m128i*)pTotalLine, (__m128i*)pToLine, mInvLeni, width);
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			OffsetPtr(pFromLine2, fromLineOffsetBytes);
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		if (bBottom) {
			pFromLine2 = pFrom;
			OffsetPtr(pFromLine2, (kye0 - r) * fromLineOffsetBytes);
			pFromLine = pFrom;
			OffsetPtr(pFromLine, (height - 1) * fromLineOffsetBytes);
			for (size_t y=kye0,cnt=0; y<height; ++y, ++cnt) {
				Worker::process((const __m128i*)pFromLine2, (const __m128i*)pFromLine, (__m128i*)pTotalLine, (__m128i*)pToLine, mInvLeni, width);
				OffsetPtr(pFromLine2, fromLineOffsetBytes);
				OffsetPtr(pFromLine, -fromLineOffsetBytes);
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}
	}
}

struct HorizontalCollector {
	
	const uint16_t width;
	const int r;
	
	const size_t kxs0;
	const size_t kxe0;
	
	HorizontalCollector(
		uint16_t width,
		int r
	)
		:
		width(width),
		r(r),
		kxs0( std::min<size_t>(width, 1 + r) ),
		kxe0( std::max<int>(0, width - r) )
	{
	}
	
	template <typename OperatorT>
	__forceinline void process(const uint8_t* pSrc, OperatorT& op) {
		const uint8_t* pSrc2 = pSrc + r;
		int total = *pSrc++;
		for (size_t kx=1; kx<=r; ++kx) {
			total += *pSrc++ * 2;
		}
		op.process(total);
		for (size_t x=1; x<=r; ++x) {
			total -= *pSrc2--;
			total += *pSrc++;
			op.process(total);
		}
		for (size_t x=r+1; x<kxe0; ++x) {
			total -= *pSrc2++;
			total += *pSrc++;
			op.process(total);
		}
		--pSrc;
		for (size_t x=kxe0; x<width; ++x) {
			total -= *pSrc2++;
			total += *--pSrc;
			op.process(total);
		}
	}
};

void test_8(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int r = std::min<int>(height, std::min<int>(width, radius));
	uint16_t* pTotalLine = (uint16_t*) pTotal;
	
	int len = r * 2 + 1; // diameter
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	// vertical
	int kys0 = 1;
	int kye0 = height;
	if (bTop) {
		kys0 = std::min<int>(height, 1 + r);
	}
	if (bBottom) {
		kye0 = std::max<int>(0, height - r);
	}
	
	HorizontalCollector horizontal(width, r);
	
	struct MovOperator {
		uint8_t* pXDest;
		uint16_t* pYTotal;
		const int invLen;
		
		MovOperator(int invLen) : invLen(invLen) {}
		
		__forceinline void process(int32_t total) {
			int v = (total * invLen) >> SHIFT;
			*pXDest++ = v;
			*pYTotal++ = v;
		}
	};

	struct AddOperator {
		uint8_t* pXDest;
		uint16_t* pYTotal;
		int invLen;
		
		AddOperator(int invLen) : invLen(invLen) {}

		__forceinline void process(int32_t total) {
			int v = (total * invLen) >> SHIFT;
			*pXDest++ = v;
			*pYTotal++ += v;
		}
	};

	struct Add2Operator {
		uint8_t* pXDest;
		uint16_t* pYTotal;
		int invLen;
		
		Add2Operator(int invLen) : invLen(invLen) {}

		__forceinline void process(int32_t total) {
			int v = total * invLen;
			*pXDest++ = v >> SHIFT;
			*pYTotal++ += v >> (SHIFT - 1);
		}
	};

	struct SlideOperator {
		uint8_t* __restrict pXDest;
		const uint8_t* __restrict pYSub;
		uint16_t* pYTotal;
		uint8_t* __restrict pYDest;
		int invLen;
		
		SlideOperator(int invLen) : invLen(invLen) {}

		__forceinline void process(int32_t total) {
			int add = (total * invLen) >> SHIFT;
			*pXDest++ = add;
			int sub = *pYSub++;
			int yTotal = *pYTotal - sub + add;
			*pYTotal++ = yTotal;
			*pYDest++ = (yTotal * invLen) >> SHIFT;
		}
	};
	
	// TODO: ï°êîâÒåJÇËï‘Ç∑èÍçáÇÕÅAì«Ç›éÊÇÁÇÍÇÈéQè∆ópÇÃóÃàÊÇÇøÇ·ÇÒÇ∆ópà”
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			pFrom = pWork2;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		const uint8_t* pFromLine = pFrom;
		uint8_t* pWorkLine = pWork;
		uint8_t* pWorkLine2 = pWorkLine;
		uint8_t* pToLine = pTo;
		
		SlideOperator slideOp(invLen);
		
		if (bTop) {
			MovOperator movOp(invLen);
			Add2Operator add2Op(invLen);
			movOp.pYTotal = pTotalLine;
			movOp.pXDest = pWorkLine;
			horizontal.process(pFromLine, movOp);
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			for (size_t ky=1; ky<=r; ++ky) {
				add2Op.pXDest = pWorkLine;
				add2Op.pYTotal = pTotalLine;
				horizontal.process(pFromLine, add2Op);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				OffsetPtr(pWorkLine, workLineOffsetBytes);
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
			OffsetPtr(pWorkLine2, r * workLineOffsetBytes);
			
			for (size_t y=1; y<=r; ++y) {
				slideOp.pYTotal = pTotalLine;
				slideOp.pXDest = pWorkLine;
				slideOp.pYSub = pWorkLine2;
				slideOp.pYDest = pToLine;
				horizontal.process(pFromLine, slideOp);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				OffsetPtr(pWorkLine2, -workLineOffsetBytes);
				OffsetPtr(pWorkLine, workLineOffsetBytes);
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}else {
			AddOperator addOp(invLen);
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] = 0;
			}
			OffsetPtr(pFromLine, -r * fromLineOffsetBytes);
			OffsetPtr(pWorkLine, -r * workLineOffsetBytes);
			pWorkLine2 = pWorkLine;
			for (int y=-r; y<=r; ++y) {
				addOp.pYTotal = pTotalLine;
				addOp.pXDest = pWorkLine;
				horizontal.process(pFromLine, addOp);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				OffsetPtr(pWorkLine, workLineOffsetBytes);
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		for (size_t y=kys0; y<kye0; ++y) {
			slideOp.pYTotal = pTotalLine;
			slideOp.pXDest = pWorkLine;
			slideOp.pYSub = pWorkLine2;
			slideOp.pYDest = pToLine;
			horizontal.process(pFromLine, slideOp);
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pWorkLine2, workLineOffsetBytes);
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		if (bBottom) {
			pWorkLine2 = pWork;
			OffsetPtr(pWorkLine2, (kye0 - r) * workLineOffsetBytes);
			pWorkLine = pWork;
			OffsetPtr(pWorkLine, (height - 1) * workLineOffsetBytes);
			for (size_t y=kye0,cnt=0; y<height; ++y, ++cnt) {
				for (size_t x=0; x<width; ++x) {
					int total = pTotalLine[x] - pWorkLine2[x] + pWorkLine[x];
					pToLine[x] = (total * invLen) >> SHIFT;
					pTotalLine[x] = total;
				}
				OffsetPtr(pWorkLine2, workLineOffsetBytes);
				OffsetPtr(pWorkLine, -workLineOffsetBytes);
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}
	}
}

void test_9(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int r = std::min<int>(height, std::min<int>(width, radius));
	uint16_t* pTotalLine = (uint16_t*) pTotal;
	
	int len = r * 2 + 1; // diameter
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	struct HorizontalCollector {
		const uint16_t width;
		const int r;
		const int invLen;
		
		const size_t kxs0;
		const size_t kxe0;
		
		HorizontalCollector(
			uint16_t width,
			int r,
			int invLen
		)
			:
			width(width),
			r(r),
			invLen(invLen),
			kxs0( std::min<size_t>(width, 1 + r) ),
			kxe0( std::max<int>(0, width - r) )
		{
		}
		
		__forceinline void process(
			const uint8_t* __restrict pSrc,
			uint8_t* __restrict pWork
		) {
			const uint8_t* pSrc2 = pSrc + r;
			int total = *pSrc++;
			for (size_t kx=1; kx<=r; ++kx) {
				total += *pSrc++ * 2;
			}
			*pWork++ = (total * invLen) >> SHIFT;
			for (size_t x=1; x<=r; ++x) {
				total -= *pSrc2--;
				total += *pSrc++;
				*pWork++ = (total * invLen) >> SHIFT;
			}
			for (size_t x=r+1; x<kxe0; ++x) {
				total -= *pSrc2++;
				total += *pSrc++;
				*pWork++ = (total * invLen) >> SHIFT;
			}
			--pSrc;
			for (size_t x=kxe0; x<width; ++x) {
				total -= *pSrc2++;
				total += *--pSrc;
				*pWork++ = (total * invLen) >> SHIFT;
			}
		}
	} horizontal(width, r, invLen);
	
	struct VerticalCollector {
		const uint16_t width;
		const int invLen;
		VerticalCollector(uint16_t width, int invLen)
			:
			width(width),
			invLen(invLen)
		{
		}
		
		__forceinline void process(
			const uint8_t* __restrict pSubLine,
			const uint8_t* __restrict pAddLine,
			uint16_t* pTotalLine,
			uint8_t* __restrict pDestLine
			)
		{
			for (size_t x=0; x<width; ++x) {
				int total = pTotalLine[x] - pSubLine[x] + pAddLine[x];
				pDestLine[x] = (total * invLen) >> SHIFT;
				pTotalLine[x] = total;
			}
		}
	} vertical(width, invLen);
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			pFrom = pWork2;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		const uint8_t* pFromLine = pFrom;
		RingLinePtr<uint8_t*> pWorkLine(len+1, 0, pWork, workLineOffsetBytes);
		RingLinePtr<uint8_t*> pWorkLine2(pWorkLine);
		uint8_t* pToLine = pTo;
		
		if (bTop) {
			horizontal.process(pFromLine, pWorkLine);
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] = pWorkLine[x];
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pWorkLine.moveNext();
			for (size_t ky=1; ky<=r; ++ky) {
				horizontal.process(pFromLine, pWorkLine);
				for (size_t x=0; x<width; ++x) {
					pTotalLine[x] += pWorkLine[x] * 2;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
			pWorkLine2.move(r);
			
			for (size_t y=1; y<=r; ++y) {
				horizontal.process(pFromLine, pWorkLine);
				vertical.process(pWorkLine2, pWorkLine, pTotalLine, pToLine);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine2.movePrev();
				pWorkLine.moveNext();
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}else {
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] = 0;
			}
			int offset = (iterationCount - n) * -r;
			OffsetPtr(pFromLine, offset * fromLineOffsetBytes);
			pWorkLine.move(offset);
			pWorkLine2.move(offset);
			OffsetPtr(pToLine, (iterationCount - n - 1) * -r * toLineOffsetBytes);
			for (int y=-r; y<=r; ++y) {
				horizontal.process(pFromLine, pWorkLine);
				for (size_t x=0; x<width; ++x) {
					pTotalLine[x] += pWorkLine[x];
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		int loopCount = height - 1;
		if (bTop) {
			loopCount -= r;
		}else {
			loopCount += (iterationCount - n - 1) * r;
		}
		if (bBottom) {
			loopCount -= r;
		}else {
			loopCount += (iterationCount - n - 1) * r;
		}
		
		for (size_t y=0; y<loopCount; ++y) {
			horizontal.process(pFromLine, pWorkLine);
			vertical.process(pWorkLine2, pWorkLine, pTotalLine, pToLine);
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pWorkLine.moveNext();
			pWorkLine2.moveNext();
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		if (bBottom) {
			pWorkLine2.move(-2);
			for (size_t y=0; y<r; ++y) {
				vertical.process(pWorkLine2, pWorkLine, pTotalLine, pToLine);
				pWorkLine2.moveNext();
				pWorkLine.movePrev();
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}
	}
}

struct HorizontalProcessor {
	const uint16_t width;
	const int r;
	const int invLen;
	const __m128i mInvLen;
	
	const size_t kxs0;
	const size_t kxe0;
	const size_t kxs0_16end;
	const size_t kxe0_16end;
	
	HorizontalProcessor(
		uint16_t width,
		int r,
		int invLen
	)
		:
		width(width),
		r(r),
		invLen(invLen),
		mInvLen(_mm_set1_epi16(invLen)),
		kxs0( std::min<size_t>(width, 1 + r) ),
		kxe0( std::max<int>(0, width - r) ),
		kxs0_16end(kxs0 + (16 - (kxs0 % 16))),
		kxe0_16end(kxe0 - (kxe0 % 16))
	{
	}
	
	__forceinline void process(
		const uint8_t* __restrict pSrc,
		uint8_t* __restrict pWork
	) {
		const uint8_t* pSrcOrg = pSrc;
		
		const uint8_t* pSrc2 = pSrc + r;
		int total = *pSrc++;
		for (size_t kx=1; kx<=r; ++kx) {
			total += *pSrc++ * 2;
		}
		*pWork++ = (total * invLen) >> SHIFT;
		for (size_t x=1; x<=r; ++x) {
			total -= *pSrc2--;
			total += *pSrc++;
			*pWork++ = (total * invLen) >> SHIFT;
		}
		for (size_t x=r+1; x<kxs0_16end; ++x) {
			total -= *pSrc2++;
			total += *pSrc++;
			*pWork++ = (total * invLen) >> SHIFT;
		}
		
		__m128i* mPSub = (__m128i*) pSrc2;
		__m128i* mPAdd = (__m128i*) pSrc;
		__m128i mNextSub = _mm_loadu_si128(mPSub++); // hoist loading
		__m128i mNextAdd = _mm_loadu_si128(mPAdd++);
		__m128i* mPWork = (__m128i*) pWork;
		
		__m128i mTotal = _mm_set1_epi16(total);
		const size_t loopCount = (kxe0_16end - kxs0_16end) / 16;
		for (size_t x=0; x<loopCount; ++x) {
			__m128i mSub = mNextSub;
			__m128i mAdd = mNextAdd;
			mNextSub = _mm_loadu_si128(mPSub++);
			mNextAdd = _mm_loadu_si128(mPAdd++);
			
			__m128i mSub0 = _mm_unpacklo_epi8(mSub, _mm_setzero_si128());
			__m128i mSub1 = _mm_unpackhi_epi8(mSub, _mm_setzero_si128());
			__m128i mAdd0 = _mm_unpacklo_epi8(mAdd, _mm_setzero_si128());
			__m128i mAdd1 = _mm_unpackhi_epi8(mAdd, _mm_setzero_si128());
			
			__m128i mDiff0 = _mm_sub_epi16(mAdd0, mSub0);
			__m128i mDiff1 = _mm_sub_epi16(mAdd1, mSub1);
			mDiff0 = _mm_add_epi16(mDiff0, _mm_slli_si128(mDiff0, 2));
			mDiff0 = _mm_add_epi16(mDiff0, _mm_slli_si128(mDiff0, 4));
			mDiff0 = _mm_add_epi16(mDiff0, _mm_slli_si128(mDiff0, 8));
			__m128i jump = _mm_shufflehi_epi16(mDiff0, _MM_SHUFFLE(3,3,3,3));
			jump = _mm_unpackhi_epi64(jump, jump);
			
			mDiff1 = _mm_add_epi16(mDiff1, _mm_slli_si128(mDiff1, 2));
			mDiff1 = _mm_add_epi16(mDiff1, _mm_slli_si128(mDiff1, 4));
			mDiff1 = _mm_add_epi16(mDiff1, _mm_slli_si128(mDiff1, 8));
			mDiff1 = _mm_add_epi16(mDiff1, jump);
			
			__m128i left = _mm_add_epi16(mTotal, mDiff0);
			__m128i right = _mm_add_epi16(mTotal, mDiff1);
			__m128i left2 = _mm_mulhi_epu16(left, mInvLen);
			__m128i right2 = _mm_mulhi_epu16(right, mInvLen);
			__m128i result = _mm_packus_epi16(left2, right2);
			*mPWork++ = result;
			mTotal = _mm_shufflehi_epi16(right, _MM_SHUFFLE(3,3,3,3));
			mTotal = _mm_unpackhi_epi64(mTotal, mTotal);
		}
		total = _mm_extract_epi16(mTotal, 0);
		const int advance = kxe0_16end - kxs0_16end;
		pSrc2 += advance;
		pSrc += advance;
		pWork += advance;
		for (size_t x=kxe0_16end; x<kxe0; ++x) {
			total -= *pSrc2++;
			total += *pSrc++;
			*pWork++ = (total * invLen) >> SHIFT;
		}
		pSrc -= 2;
		for (size_t x=kxe0; x<width; ++x) {
			total -= *pSrc2++;
			total += *pSrc--;
			*pWork++ = (total * invLen) >> SHIFT;
		}
	}
};

struct VerticalProcessor {
	const uint16_t width;
	const int invLen;
	const __m128i mInvLen;
	const size_t loopCount;
	
	VerticalProcessor(
		uint16_t width,
		int invLen
	)
		:
		width(width),
		invLen(invLen),
		mInvLen(_mm_set1_epi16(invLen)),
		loopCount(width >> 4)
	{
	}
	
	__forceinline void process(
		const __m128i* __restrict pSubLine,
		const __m128i* __restrict pAddLine,
		__m128i* __restrict pTotalLine,
		__m128i* __restrict pDestLine
	) {
		for (size_t i=0; i<loopCount; ++i) {
			__m128i mSub = pSubLine[i];
			__m128i mAdd = pAddLine[i];
			
			__m128i mTotal0 = pTotalLine[i*2+0];
			__m128i mTotal1 = pTotalLine[i*2+1];
			
			__m128i mSub0 = _mm_unpacklo_epi8(mSub, _mm_setzero_si128());
			__m128i mSub1 = _mm_unpackhi_epi8(mSub, _mm_setzero_si128());
			__m128i mAdd0 = _mm_unpacklo_epi8(mAdd, _mm_setzero_si128());
			__m128i mAdd1 = _mm_unpackhi_epi8(mAdd, _mm_setzero_si128());
			
			__m128i mDiff0 = _mm_sub_epi16(mAdd0, mSub0);
			__m128i mDiff1 = _mm_sub_epi16(mAdd1, mSub1);
			
			mTotal0 = _mm_add_epi16(mTotal0, mDiff0);
			mTotal1 = _mm_add_epi16(mTotal1, mDiff1);
			
			pTotalLine[i*2+0] = mTotal0;
			pTotalLine[i*2+1] = mTotal1;
			
			__m128i mDest0 = _mm_mulhi_epu16(mTotal0, mInvLen);
			__m128i mDest1 = _mm_mulhi_epu16(mTotal1, mInvLen);
			
			__m128i mResult = _mm_packus_epi16(mDest0, mDest1);
			
//			_mm_stream_si128(pDestLine+i, mResult);	// for single pass
			pDestLine[i] = mResult;	// for multi pass
		}
		const size_t remainLoopCount = width & 0xF;
		uint8_t* pDest = (uint8_t*) (pDestLine + loopCount);
		uint8_t* pSub = (uint8_t*) (pSubLine + loopCount);
		uint8_t* pAdd = (uint8_t*) (pAddLine + loopCount);
		int16_t* pTotal = (int16_t*) (pTotalLine + loopCount * 2);
		int16_t invLen = _mm_extract_epi16(mInvLen, 0);				
		for (size_t i=0; i<remainLoopCount; ++i) {
			int16_t total = pTotal[i];
			total += pAdd[i] - pSub[i];
			pDest[i] = ((total * invLen) >> SHIFT);
			pTotal[i] = total;
		}
	}
};

void test_10(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int r = std::min<int>(height, std::min<int>(width, radius));
	uint16_t* pTotalLine = (uint16_t*) pTotal;
	
	int len = r * 2 + 1; // diameter
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFrom;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFrom = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			pFrom = pWork2;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pTo;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pTo = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pTo = pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		const uint8_t* pFromLine = pFrom;
		RingLinePtr<uint8_t*> pWorkLine(len+1, 0, pWork, workLineOffsetBytes);
		RingLinePtr<uint8_t*> pWorkLine2(pWorkLine);
		uint8_t* pToLine = pTo;
		
		HorizontalProcessor horizontal(width, r, invLen);
		VerticalProcessor vertical(width, invLen);
		
		if (bTop) {
			horizontal.process(pFromLine, pWorkLine);
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] = pWorkLine[x];
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pWorkLine.moveNext();
			for (size_t ky=1; ky<=r; ++ky) {
				horizontal.process(pFromLine, pWorkLine);
				const __m128i* pMWork = (const __m128i*)pWorkLine;
				__m128i* pMTotal = (__m128i*)pTotalLine;
				for (size_t x=0; x<width>>4; ++x) {
					__m128i mData = pMWork[x];
					__m128i mLeft = _mm_unpacklo_epi8(mData, _mm_setzero_si128());
					__m128i mRight = _mm_unpackhi_epi8(mData, _mm_setzero_si128());
					mLeft = _mm_add_epi16(mLeft, mLeft);
					mRight = _mm_add_epi16(mRight, mRight);
					__m128i totalLeft = *pMTotal;
					__m128i totalRight = *(pMTotal+1);
					*pMTotal++ = _mm_add_epi16(totalLeft, mLeft);
					*pMTotal++ = _mm_add_epi16(totalRight, mRight);
				}
				for (size_t x=width&0xFFF0; x<width; ++x) {
					pTotalLine[x] += pWorkLine[x] * 2;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
			pWorkLine2.move(r);
			
			for (size_t y=1; y<=r; ++y) {
				horizontal.process(pFromLine, pWorkLine);
				vertical.process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pToLine);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine2.movePrev();
				pWorkLine.moveNext();
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}else {
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] = 0;
			}
			int offset = (iterationCount - n) * -r;
			OffsetPtr(pFromLine, offset * fromLineOffsetBytes);
			pWorkLine.move(offset);
			pWorkLine2.move(offset);
			OffsetPtr(pToLine, (iterationCount - n - 1) * -r * toLineOffsetBytes);
			for (int y=-r; y<=r; ++y) {
				horizontal.process(pFromLine, pWorkLine);
				const __m128i* pMWork = (const __m128i*)pWorkLine;
				__m128i* pMTotal = (__m128i*)pTotalLine;
				for (size_t x=0; x<width>>4; ++x) {
					__m128i mData = pMWork[x];
					__m128i mLeft = _mm_unpacklo_epi8(mData, _mm_setzero_si128());
					__m128i mRight = _mm_unpackhi_epi8(mData, _mm_setzero_si128());
					
					__m128i totalLeft = *pMTotal;
					__m128i totalRight = *(pMTotal+1);
					*pMTotal++ = _mm_add_epi16(totalLeft, mLeft);
					*pMTotal++ = _mm_add_epi16(totalRight, mRight);
				}
				for (size_t x=width&0xFFF0; x<width; ++x) {
					pTotalLine[x] += pWorkLine[x];
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		int loopCount = height - 1;
		if (bTop) {
			loopCount -= r;
		}else {
			loopCount += (iterationCount - n - 1) * r;
		}
		if (bBottom) {
			loopCount -= r;
		}else {
			loopCount += (iterationCount - n - 1) * r;
		}
		for (size_t y=0; y<loopCount; ++y) {
			horizontal.process(pFromLine, pWorkLine);
			vertical.process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pToLine);
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pWorkLine.moveNext();
			pWorkLine2.moveNext();
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		if (bBottom) {
			pWorkLine.move(-2);
			for (size_t y=0; y<r; ++y) {
				vertical.process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pToLine);
				pWorkLine2.moveNext();
				pWorkLine.movePrev();
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}
	}
	_mm_mfence();
}

void test_11(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;

	uint32_t hRad = p.radius;
	uint32_t vRad = p.radius;
	uint32_t hLen = 1 + hRad*2;
	uint32_t vLen = 1 + vRad*2;
	uint32_t invLen = 0xFFFFFF / (hLen*vLen);
	uint32_t hCount = p.width;
	uint32_t vCount = p.height;

	const uint8_t* hLine = p.pSrc;
	uint8_t* vLine = p.pDest;
	OffsetPtr(vLine, destLineOffsetBytes * vRad);

	uint32_t* vSumLine = (uint32_t*)pWork2;
	RingLinePtr<uint16_t*> vMinusLine(vLen, 0, (uint16_t*)pWork, width*2);
	RingLinePtr<uint16_t*> vPlusLine(vLen, 0, (uint16_t*)pWork, width*2);

	// vTop collect
	for (size_t y=0; y<vLen; ++y) {
		const uint8_t* hMinus = hLine;
		const uint8_t* hPlus = hLine+hLen;
		size_t hSum = 0;
		// hLeft collect
		for (size_t x=0; x<hLen; ++x) {
			hSum += hLine[x];
		}
		// hCenter
		for (size_t x=hRad; x<hCount-hRad; ++x) {
			hSum -= *hMinus++;
			hSum += *hPlus++;
			vPlusLine[x] = hSum;
			vSumLine[x] += hSum;
		}
		// hRight
		;
		OffsetPtr(hLine, srcLineOffsetBytes);
		vPlusLine.moveNext();
	}

	// vMiddle
	for (size_t y=vRad; y<vCount-vLen; ++y) {

		const uint8_t* hMinus = hLine;
		const uint8_t* hPlus = hLine+hLen;
		size_t hSum = 0;
		// hLeft collect
		for (size_t x=0; x<hLen; ++x) {
			hSum += hLine[x];
		}
		// hCenter
		for (size_t x=hRad; x<hCount-hRad; ++x) {
			hSum -= *hMinus++;
			hSum += *hPlus++;
			
			// in this way, vPlus memory read is not required.
			// but memory access pattern is a bit complex.
			uint32_t vSum = vSumLine[x];
			vSum -= vMinusLine[x];
			vSum += hSum;
			vPlusLine[x] = hSum;
			vSumLine[x] = vSum;
			vLine[x] = (vSum * invLen) >> 24;
		}
		// hRight
		;
		OffsetPtr(hLine, srcLineOffsetBytes);
		OffsetPtr(vLine, destLineOffsetBytes);
		vMinusLine.moveNext();
		vPlusLine.moveNext();
	}

}

static __forceinline
void 
repeatShiftSum3(__m128i m01, __m128i& m0, __m128i& m1, __m128i& remain0)
{
	m0 = _mm_unpacklo_epi8(m01, _mm_setzero_si128());
	m1 = _mm_unpackhi_epi8(m01, _mm_setzero_si128());
#if 1
	remain0 = _mm_srli_si128(_mm_add_epi16(m1, _mm_srli_si128(m1, 2)), 12);

	__m128i m01_1 = _mm_slli_si128(m01, 1);
	__m128i m01_2 = _mm_slli_si128(m01, 2);
	m0 = 
		_mm_add_epi16(
			_mm_add_epi16(
				m0,
				_mm_unpacklo_epi8(m01_1, _mm_setzero_si128())
			),
			_mm_unpacklo_epi8(m01_2, _mm_setzero_si128())
		);
	m1 = 
		_mm_add_epi16(
			_mm_add_epi16(
				m1,
				_mm_unpackhi_epi8(m01_1, _mm_setzero_si128())
			),
			_mm_unpackhi_epi8(m01_2, _mm_setzero_si128())
		);
#else
	__m128i s0L = _mm_add_epi16(_mm_add_epi16(m0, _mm_slli_si128(m0, 2)), _mm_slli_si128(m0, 4));
	__m128i s1L = _mm_add_epi16(_mm_add_epi16(m1, _mm_slli_si128(m1, 2)), _mm_slli_si128(m1, 4));
	__m128i s0R = _mm_srli_si128(_mm_add_epi16(m0, _mm_srli_si128(m0, 2)), 12);
	__m128i s1R = _mm_srli_si128(_mm_add_epi16(m1, _mm_srli_si128(m1, 2)), 12);

	m0 = s0L;
	m1 = _mm_add_epi16(s1L, s0R);
	remain0 = s1R;
#endif
}

#if _MSC_VER >= 1700

static __forceinline
void repeatShiftSum3(__m256i m01, __m256i& m0, __m256i& m1, __m256i& remain0)
{
	m0 = _mm256_unpacklo_epi8(m01, _mm256_setzero_si256());
	m1 = _mm256_unpackhi_epi8(m01, _mm256_setzero_si256());
	remain0 = _mm256_srli_si256(_mm256_add_epi16(m1, _mm256_srli_si256(m1, 2)), 28);

	__m256i m01_1 = _mm256_slli_si256(m01, 1);
	__m256i m01_2 = _mm256_slli_si256(m01, 2);
	m0 = 
		_mm256_add_epi16(
			_mm256_add_epi16(
				m0,
				_mm256_unpacklo_epi8(m01_1, _mm256_setzero_si256())
			),
			_mm256_unpacklo_epi8(m01_2, _mm256_setzero_si256())
		);
	m1 = 
		_mm256_add_epi16(
			_mm256_add_epi16(
				m1,
				_mm256_unpackhi_epi8(m01_1, _mm256_setzero_si256())
			),
			_mm256_unpackhi_epi8(m01_2, _mm256_setzero_si256())
		);

}

#endif

template <size_t SHIFTS>
static __forceinline
void repeatShiftSum(__m128i main01, __m128i& main0, __m128i& main1, __m128i& remain0)
{
	
	// TODO: efficiently construct slided sum
	// 2 ^ 2 ^ 2 = 8
	main0 = _mm_unpacklo_epi8(main01, _mm_setzero_si128());
	main1 = _mm_unpackhi_epi8(main01, _mm_setzero_si128());

	__m128i m0L = main0;
	__m128i m0R = main0;
	__m128i m1L = main1;
	__m128i m1R = main1;
	
	__m128i s0L = main0;
	__m128i s0R = main0;
	__m128i s1L = main1;
	__m128i s1R = main1;

	for (size_t i=0; i<SHIFTS-1; ++i) {
		m0L = _mm_slli_si128(m0L, 2);
		m0R = _mm_srli_si128(m0R, 2);
		m1L = _mm_slli_si128(m1L, 2);
		m1R = _mm_srli_si128(m1R, 2);
		s0L = _mm_add_epi16(s0L, m0L);
		s0R = _mm_add_epi16(s0R, m0R);
		s1L = _mm_add_epi16(s1L, m1L);
		s1R = _mm_add_epi16(s1R, m1R);
	}
	
	main0 = s0L;
	s0R = _mm_srli_si128(s0R, (8-(SHIFTS-1))*2);
	main1 = _mm_add_epi16(s1L, s0R);
	remain0 = _mm_srli_si128(s1R, (8-(SHIFTS-1))*2);;
}

static __forceinline
void repeatShiftNum(__m128i main01, __m128i& main0, __m128i& main1, __m128i& remain0, size_t count)
{
	switch (count) {
	case 3:
		repeatShiftSum3(main01, main0, main1, remain0);
		break;
	case 5:
		repeatShiftSum<5>(main01, main0, main1, remain0);
		break;
	case 7:
		repeatShiftSum<7>(main01, main0, main1, remain0);
		break;
	case 9:
		repeatShiftSum<9>(main01, main0, main1, remain0);
		break;
	default:
		break;
	}
}

void test_12(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;

	uint32_t hRad = p.radius;
	uint32_t vRad = p.radius;
	uint32_t hLen = 1 + hRad*2;
	uint32_t vLen = 1 + vRad*2;
	uint32_t invLen = 0xFFFFFF / (hLen*vLen);
	uint32_t hCount = p.width;
	uint32_t vCount = p.height;

	const __m128i mInvRatio = _mm_set1_epi16(0xFFFF / (hLen*vLen));
	
	if (hLen > 9) {
		return;
	}
	
	const uint8_t* hLine = p.pSrc;
	uint8_t* vLine = p.pDest;
	OffsetPtr(vLine, destLineOffsetBytes * vRad);

	uint16_t* vSumLine = (uint16_t*)pWork2;
	assert((ptrdiff_t)vSumLine % 16 == 0);
	assert((width * 2) % 16 == 0);

	RingLinePtr<uint16_t*> vMinusLine(vLen, 0, (uint16_t*)pWork, width*2);
	RingLinePtr<uint16_t*> vPlusLine(vLen, 0, (uint16_t*)pWork, width*2);

	// vTop collect
	{
		const uint8_t* hMinus = hLine;
		const uint8_t* hPlus = hLine+hLen;
		size_t hSum = 0;
		// hLeft collect
		for (size_t x=0; x<hLen; ++x) {
			hSum += hLine[x];
		}
		// hCenter
		for (size_t x=hRad; x<hCount-hRad; ++x) {
			hSum -= *hMinus++;
			hSum += *hPlus++;
			vPlusLine[x] = hSum;
			vSumLine[x] = hSum;
		}
		// hRight
		;
		OffsetPtr(hLine, srcLineOffsetBytes);
		vPlusLine.moveNext();
	}
	for (size_t y=1; y<vLen; ++y) {
		const uint8_t* hMinus = hLine;
		const uint8_t* hPlus = hLine+hLen;
		size_t hSum = 0;
		// hLeft collect
		for (size_t x=0; x<hLen; ++x) {
			hSum += hLine[x];
		}
		// hCenter
		for (size_t x=hRad; x<hCount-hRad; ++x) {
			hSum -= *hMinus++;
			hSum += *hPlus++;
			vPlusLine[x] = hSum;
			vSumLine[x] += hSum;
		}
		// hRight
		;
		OffsetPtr(hLine, srcLineOffsetBytes);
		vPlusLine.moveNext();
	}

	__m128i* mvSumLine = (__m128i*)vSumLine;

	// vMiddle
	for (size_t y=vRad; y<vCount-vLen; ++y) {

		assert((ptrdiff_t)hLine % 16 == 0);

		const __m128i* mhLine = (const __m128i*)hLine;
		__m128i m01 = mhLine[0];
		__m128i m0 = _mm_unpacklo_epi8(m01, _mm_setzero_si128());
		__m128i m1 = _mm_unpackhi_epi8(m01, _mm_setzero_si128());
		__m128i m23 = mhLine[1];

		// hLeft collect
		__m128i m1r;

		__m128i* mvLine = (__m128i*) vLine;
		__m128i* mvMinusLine = vMinusLine;
		__m128i* mvPlusLine = vMinusLine;
		__m128i m0l = _mm_setzero_si128();
		
		// hCenter
		const size_t loopCount = hCount / 16;
		for (size_t i=0; i<loopCount; ++i) {
			repeatShiftNum(mhLine[i], m0, m1, m1r, hLen);
			m0 = _mm_add_epi16(m0l, m0);
			__m128i sum0 = mvSumLine[i*2+0];
			__m128i sum1 = mvSumLine[i*2+1];
			__m128i minus0 = mvMinusLine[i*2+0];
			__m128i minus1 = mvMinusLine[i*2+1];
			sum0 = _mm_sub_epi16(sum0, minus0);
			sum1 = _mm_sub_epi16(sum1, minus1);
			mvPlusLine[i*2+0] = m0;
			mvPlusLine[i*2+1] = m1;
			sum0 = _mm_add_epi16(sum0, m0);
			sum1 = _mm_add_epi16(sum1, m1);
			mvSumLine[i*2+0] = sum0;
			mvSumLine[i*2+1] = sum1;

			_mm_stream_si128(mvLine+i, _mm_packus_epi16(
				_mm_mulhi_epu16(sum0, mInvRatio),
				_mm_mulhi_epu16(sum1, mInvRatio)
				));
			m0l = m1r;
		}
		// hRight
		;
		OffsetPtr(hLine, srcLineOffsetBytes);
		OffsetPtr(vLine, destLineOffsetBytes);
		vMinusLine.moveNext();
		vPlusLine.moveNext();
	}

}

void test_13(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;

	uint32_t hRad = p.radius;
	uint32_t vRad = p.radius;
	uint32_t hLen = 1 + hRad*2;
	uint32_t vLen = 1 + vRad*2;
	uint32_t invLen = 0xFFFFFF / (hLen*vLen);
	uint32_t hCount = p.width;
	uint32_t vCount = p.height;

	static const __m128i mInvRatio = _mm_set1_epi16(0xFFFF / 9);
	
	if (vRad != 1) {
		return;
	}
	
	const uint8_t* hLine = p.pSrc;
	uint8_t* vLine = p.pDest;
	OffsetPtr(vLine, destLineOffsetBytes * vRad);

	uint16_t* vSumLine = (uint16_t*)pWork2;
	assert((ptrdiff_t)vSumLine % 16 == 0);
	assert((width * 2) % 16 == 0);

	int* remains = (int*)p.pWork;
#if 1
	const __m128i* mpSrc = (const __m128i*)pSrc;
	__m128i* mpDst = (__m128i*)pDest;
	for (size_t i=0; i<hCount/64; ++i) {
		const __m128i* src = mpSrc;
		__m128i* dst = mpDst;
		__m128i sums0 = _mm_setzero_si128();
		__m128i sums1 = _mm_setzero_si128();
		__m128i sums2 = _mm_setzero_si128();
		__m128i sums3 = _mm_setzero_si128();
		__m128i sums4 = _mm_setzero_si128();
		__m128i sums5 = _mm_setzero_si128();
		__m128i sums6 = _mm_setzero_si128();
		__m128i sums7 = _mm_setzero_si128();
		__m128i adds0 = _mm_setzero_si128();
		__m128i adds1 = _mm_setzero_si128();
		__m128i adds2 = _mm_setzero_si128();
		__m128i adds3 = _mm_setzero_si128();
		__m128i adds4 = _mm_setzero_si128();
		__m128i adds5 = _mm_setzero_si128();
		__m128i adds6 = _mm_setzero_si128();
		__m128i adds7 = _mm_setzero_si128();
		__m128i mids0 = _mm_setzero_si128();
		__m128i mids1 = _mm_setzero_si128();
		__m128i mids2 = _mm_setzero_si128();
		__m128i mids3 = _mm_setzero_si128();
		__m128i mids4 = _mm_setzero_si128();
		__m128i mids5 = _mm_setzero_si128();
		__m128i mids6 = _mm_setzero_si128();
		__m128i mids7 = _mm_setzero_si128();
		__m128i remain0 = _mm_setzero_si128();
		__m128i remain1 = _mm_setzero_si128();
		__m128i remain2 = _mm_setzero_si128();
		__m128i nsrc0 = src[0];
		__m128i nsrc1 = src[1];
		__m128i nsrc2 = src[2];
		__m128i nsrc3 = src[3];
		for (size_t y=0; y<vCount; ++y) {
			sums0 = _mm_sub_epi16(sums0, mids0);
			sums1 = _mm_sub_epi16(sums1, mids1);
			sums2 = _mm_sub_epi16(sums2, mids2);
			sums3 = _mm_sub_epi16(sums3, mids3);
			sums4 = _mm_sub_epi16(sums4, mids4);
			sums5 = _mm_sub_epi16(sums5, mids5);
			sums6 = _mm_sub_epi16(sums6, mids6);
			sums7 = _mm_sub_epi16(sums7, mids7);

			mids0 = adds0;
			mids1 = adds1;
			mids2 = adds2;
			mids3 = adds3;
			mids4 = adds4;
			mids5 = adds5;
			mids6 = adds6;
			mids7 = adds7;

			__m128i src0 = nsrc0;
			__m128i src1 = nsrc1;
			__m128i src2 = nsrc2;
			__m128i src3 = nsrc3;
			nsrc0 = src[4];
			nsrc1 = src[5];
			nsrc2 = src[6];
			nsrc3 = src[7];

			__m128i remain = _mm_cvtsi32_si128(remains[y]);
			repeatShiftSum3(src0, adds0, adds1, remain0);
			adds0 = _mm_add_epi16(adds0, remain);
			repeatShiftSum3(src1, adds2, adds3, remain1);
			repeatShiftSum3(src2, adds4, adds5, remain2);
			repeatShiftSum3(src3, adds6, adds7, remain);
			remains[y] = _mm_cvtsi128_si32(remain);
			adds2 = _mm_add_epi16(adds2, remain0);
			adds4 = _mm_add_epi16(adds4, remain1);
			adds6 = _mm_add_epi16(adds6, remain2);
			
			sums0 = _mm_add_epi16(sums0, adds0);
			sums1 = _mm_add_epi16(sums1, adds1);
			sums2 = _mm_add_epi16(sums2, adds2);
			sums3 = _mm_add_epi16(sums3, adds3);
			sums4 = _mm_add_epi16(sums4, adds4);
			sums5 = _mm_add_epi16(sums5, adds5);
			sums6 = _mm_add_epi16(sums6, adds6);
			sums7 = _mm_add_epi16(sums7, adds7);

			__m128i result0 = _mm_packus_epi16(_mm_mulhi_epu16(sums0, mInvRatio), _mm_mulhi_epu16(sums1, mInvRatio));
			__m128i result1 = _mm_packus_epi16(_mm_mulhi_epu16(sums2, mInvRatio), _mm_mulhi_epu16(sums3, mInvRatio));
			__m128i result2 = _mm_packus_epi16(_mm_mulhi_epu16(sums4, mInvRatio), _mm_mulhi_epu16(sums5, mInvRatio));
			__m128i result3 = _mm_packus_epi16(_mm_mulhi_epu16(sums6, mInvRatio), _mm_mulhi_epu16(sums7, mInvRatio));
			_mm_stream_si128(dst+0, result0);
			_mm_stream_si128(dst+1, result1);
			_mm_stream_si128(dst+2, result2);
			_mm_stream_si128(dst+3, result3);
			OffsetPtr(dst, destLineOffsetBytes);
			OffsetPtr(src, destLineOffsetBytes);
		}
		mpSrc += 4;
		mpDst += 4;
	}
#else
	memcpy(pDest, pSrc, vCount*hCount);
	//__m256i* __restrict mpSrc = (__m256i* __restrict)&pSrc[0];
	//__m256i* __restrict mpDst = (__m256i* __restrict)&pDest[0];
	//for (size_t i=0; i<vCount*hCount/64; ++i) {
	//	_mm256_stream_si256(&mpDst[i*2+0], _mm256_load_si256(&mpSrc[i*2+0]));
	//	_mm256_stream_si256(&mpDst[i*2+1], _mm256_load_si256(&mpSrc[i*2+1]));
	//}
#endif

}

void test_14(const Parameter& p) {
#if _MSC_VER >= 1700
	
	BLUR_EXTRACT_PARAMS;

	uint32_t hRad = p.radius;
	uint32_t vRad = p.radius;
	uint32_t hLen = 1 + hRad*2;
	uint32_t vLen = 1 + vRad*2;
	uint32_t invLen = 0xFFFFFF / (hLen*vLen);
	uint32_t hCount = p.width;
	uint32_t vCount = p.height;

	_mm256_zeroall();

	static const __m256i mInvRatio = _mm256_set1_epi16(0xFFFF / 9);
	
	if (vRad != 1) {
		return;
	}
	
	const uint8_t* hLine = p.pSrc;
	uint8_t* vLine = p.pDest;
	OffsetPtr(vLine, destLineOffsetBytes * vRad);

	uint16_t* vSumLine = (uint16_t*)pWork2;
	assert((ptrdiff_t)vSumLine % 16 == 0);
	assert((width * 2) % 16 == 0);

	int* remains = (int*)p.pWork;
	const __m256i* mpSrc = (const __m256i*)pSrc;
	__m256i* mpDst = (__m256i*)pDest;
	for (size_t i=0; i<width/64; ++i) {
		const __m256i* src = mpSrc;
		__m256i* dst = mpDst;
		__m256i sums0 = _mm256_setzero_si256();
		__m256i sums1 = _mm256_setzero_si256();
		__m256i sums2 = _mm256_setzero_si256();
		__m256i sums3 = _mm256_setzero_si256();
		__m256i adds0 = _mm256_setzero_si256();
		__m256i adds1 = _mm256_setzero_si256();
		__m256i adds2 = _mm256_setzero_si256();
		__m256i adds3 = _mm256_setzero_si256();
		__m256i mids0 = _mm256_setzero_si256();
		__m256i mids1 = _mm256_setzero_si256();
		__m256i mids2 = _mm256_setzero_si256();
		__m256i mids3 = _mm256_setzero_si256();
		__m256i remain0 = _mm256_setzero_si256();
		__m256i remain1 = _mm256_setzero_si256();
		__m256i remain2 = _mm256_setzero_si256();
		__m256i nsrc0 = src[0];
		__m256i nsrc1 = src[1];
		for (size_t y=0; y<vCount; ++y) {
			sums0 = _mm256_sub_epi16(sums0, mids0);
			sums1 = _mm256_sub_epi16(sums1, mids1);
			sums2 = _mm256_sub_epi16(sums2, mids2);
			sums3 = _mm256_sub_epi16(sums3, mids3);

			mids0 = adds0;
			mids1 = adds1;
			mids2 = adds2;
			mids3 = adds3;

			__m256i src0 = nsrc0;
			__m256i src1 = nsrc1;
			nsrc0 = src[2];
			nsrc1 = src[3];

			__m256i remain = _mm256_castsi128_si256(_mm_cvtsi32_si128(remains[y]));
			repeatShiftSum3(src0, adds0, adds1, remain0);
			adds0 = _mm256_add_epi16(adds0, remain);
			repeatShiftSum3(src1, adds2, adds3, remain1);
			remains[y] = _mm_cvtsi128_si32(_mm256_castsi256_si128(remain));
			adds2 = _mm256_add_epi16(adds2, remain0);
			
			sums0 = _mm256_add_epi16(sums0, adds0);
			sums1 = _mm256_add_epi16(sums1, adds1);
			sums2 = _mm256_add_epi16(sums2, adds2);
			sums3 = _mm256_add_epi16(sums3, adds3);

			__m256i result0 = _mm256_packus_epi16(_mm256_mulhi_epu16(sums0, mInvRatio), _mm256_mulhi_epu16(sums1, mInvRatio));
			__m256i result1 = _mm256_packus_epi16(_mm256_mulhi_epu16(sums2, mInvRatio), _mm256_mulhi_epu16(sums3, mInvRatio));
#if 1
			_mm256_stream_si256(dst+0, result0);
			_mm256_stream_si256(dst+1, result1);
#else
			dst[0] = result0;
			dst[1] = result1;
#endif
			OffsetPtr(dst, destLineOffsetBytes);
			OffsetPtr(src, destLineOffsetBytes);
		}
		mpSrc += 2;
		mpDst += 2;
	}
#endif
}

__forceinline
__m128i shiftAdd16(__m128i v)
{
#if 1
	v = _mm_add_epi16(v, _mm_slli_si128(v, 2));
	v = _mm_add_epi16(v, _mm_slli_si128(v, 4));
	v = _mm_add_epi16(v, _mm_slli_si128(v, 8));
#else

#endif
	return v;
}

__forceinline
void hProcess(const __m128i* src, __m128i* dst, size_t width, uint8_t len)
{
	// åªç›ÇÃé¿ëïÇ…ÇÕêßå¿óLÇË
	if (len > 7 || len == 0) {
		return;
	}
	size_t vcnt = width / 16;
	if (vcnt == 0) {
		return;
	}
	// ç∂ë§âÊñ äOÇÃóvëfÇîΩì]ÇµÇƒê∂ê¨
	__m128i prev;
	__m128i cur;
	__m128i sum0 = _mm_setzero_si128();
	
	cur = src[0];	// 0-15
	const __m128i REVERSE = _mm_setr_epi8(-1,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
	prev = _mm_shuffle_epi8(cur, REVERSE);	// 15-1

	// shift masks
	static const __m128i M[7*2] = {
		// 1
		{13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,},	// prev
		{-1,-1,-1,+0,+1,+2,+3,+4,+5,+6,+7,+8,+9,10,11,12,},	// cur
		// 2
		{11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,},	// prev
		{-1,-1,-1,-1,-1,+0,+1,+2,+3,+4,+5,+6,+7,+8,+9,10,},	// cur
		// 3
		{+9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,},	// prev
		{-1,-1,-1,-1,-1,-1,-1,+0,+1,+2,+3,+4,+5,+6,+7,+8,},	// cur
		// 4
		{+7,+8,+9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,},	// prev
		{-1,-1,-1,-1,-1,-1,-1,-1,-1,+0,+1,+2,+3,+4,+5,+6,},	// cur
		// 5
		{+5,+6,+7,+8,+9,10,11,12,13,14,15,-1,-1,-1,-1,-1,},	// prev
		{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,+0,+1,+2,+3,+4,},	// cur
		// 6
		{+3,+4,+5,+6,+7,+8,+9,10,11,12,13,14,15,-1,-1,-1,},	// prev
		{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,+0,+1,+2,},	// cur
		// 7
		{+1,+2,+3,+4,+5,+6,+7,+8,+9,10,11,12,13,14,15,-1,},	// prev
		{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,+0,},	// cur
	};
	const size_t baseIdx = (len - 1) * 2;
	const __m128i prevMask = M[baseIdx+0];
	const __m128i curMask = M[baseIdx+1];

	const __m128i invLen =  _mm_set1_epi16(0xFFFF / (1+len*2));
	const __m128i mask7 = {
		14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
	};

	uint16_t sum = 0;
	for (size_t i=0; i<1+len*2; ++i) {
		sum += ((const uint8_t*)src)[1+i];
	}
	sum0 = _mm_set1_epi16(sum);

	for (size_t i=0; i<vcnt; ++i) {
		__m128i next = src[i+1];
		__m128i prev2 = _mm_shuffle_epi8(prev, prevMask);
		__m128i cur2 = _mm_shuffle_epi8(cur, curMask);
		__m128i minus = _mm_or_si128(prev2, cur2);
		
		__m128i plus0 = _mm_unpacklo_epi8(cur, _mm_setzero_si128());
		__m128i plus1 = _mm_unpackhi_epi8(cur, _mm_setzero_si128());
		__m128i minus0 = _mm_unpacklo_epi8(minus, _mm_setzero_si128());
		__m128i minus1 = _mm_unpackhi_epi8(minus, _mm_setzero_si128());
		__m128i diff0 = _mm_sub_epi16(plus0, minus0);
		__m128i diff1 = _mm_sub_epi16(plus1, minus1);

		diff0 = shiftAdd16(diff0);
		sum0 = _mm_add_epi16(sum0, diff0);

		__m128i sum1 = _mm_shuffle_epi8(sum0, mask7);
		diff1 = shiftAdd16(diff1);
		sum1 = _mm_add_epi16(sum1, diff1);

		__m128i bytes = _mm_packus_epi16(_mm_mulhi_epu16(sum0, invLen), _mm_mulhi_epu16(sum1, invLen));
//		_mm_stream_si128(dst+i, bytes);
		dst[i] = bytes;
//		dst[i] = src[i];
		prev = cur;
		cur = next;
		sum0 = _mm_shuffle_epi8(sum1, mask7);
	}

}

/*
#if 0
		__m128i sum0 = _mm_shufflehi_epi16(sum1, _MM_SHUFFLE(3,3,3,3));
		sum0 = _mm_unpackhi_epi64(sum0, sum0);
#else
		__m128i sum0 = _mm_shuffle_epi8(sum1, MASK7);
#endif

*/

__forceinline
void hProcess2(const __m128i* src, __m128i* dst, size_t width, uint8_t len)
{
	// åªç›ÇÃé¿ëïÇ…ÇÕêßå¿óLÇË
	if (len > 14 || len == 0) {
		return;
	}
	size_t vcnt = width / 16;
	if (vcnt == 0) {
		return;
	}

	static const __m128i MASK7 = {
		14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
	};
	const __m128i REVERSE = _mm_setr_epi8(-1,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
	const __m128i invLen =  _mm_set1_epi16(0xFFFF / (1+len*2));

	// îΩì]ÇµÇƒóvëfÇê∂ê¨
	__m128i minusSrc2[2];
	minusSrc2[0] = _mm_shuffle_epi8(src[0], REVERSE);	// 15-1
	minusSrc2[1] = src[0];
	uint16_t sum = 0;
	for (size_t i=0; i<1+len*2; ++i) {
		sum += ((const uint8_t*)(minusSrc2+1)) [i - (len+1)];
	}
	__m128i sum1 = _mm_set1_epi16(sum);
	const __m128i* plusSrc = (const __m128i*) ((const uint8_t*)src + len);
	const __m128i* minusSrc = (const __m128i*) ((const uint8_t*)src - (len+1));
	__m128i plus = _mm_loadu_si128(plusSrc);
	__m128i minus = _mm_loadu_si128(
		(const __m128i*) (
			(const uint8_t*)(minusSrc2+1) - (len+1)
		)
	);

	for (size_t i=0; i<vcnt; ++i) {
		__m128i nextPlus = _mm_loadu_si128(plusSrc+i+1);
		__m128i nextMinus = _mm_loadu_si128(minusSrc+i+1);

		__m128i sum0 = _mm_shuffle_epi8(sum1, MASK7);
		__m128i plus0 = _mm_unpacklo_epi8(plus, _mm_setzero_si128());
		__m128i minus0 = _mm_unpacklo_epi8(minus, _mm_setzero_si128());
		__m128i plus1 = _mm_unpackhi_epi8(plus, _mm_setzero_si128());
		__m128i minus1 = _mm_unpackhi_epi8(minus, _mm_setzero_si128());
		__m128i diff0 = _mm_sub_epi16(plus0, minus0);
		__m128i diff1 = _mm_sub_epi16(plus1, minus1);

		diff0 = shiftAdd16(diff0);
		sum0 = _mm_add_epi16(sum0, diff0);
		sum1 = _mm_shuffle_epi8(sum0, MASK7);
		diff1 = shiftAdd16(diff1);
		sum1 = _mm_add_epi16(sum1, diff1);

		__m128i bytes = _mm_packus_epi16(_mm_mulhi_epu16(sum0, invLen), _mm_mulhi_epu16(sum1, invLen));
		dst[i] = bytes;

		plus = nextPlus;
		minus = nextMinus;

	}

}

void test_15(const Parameter& p) {
	
	const __m128i* pSrc = (const __m128i*) p.pSrc;
	__m128i* pDst = (__m128i*) p.pDest;
	for (size_t y=0; y<p.height; ++y) {
		hProcess2(pSrc, pDst, p.width, p.radius);
		OffsetPtr(pSrc, p.srcLineOffsetBytes);
		OffsetPtr(pDst, p.destLineOffsetBytes);
	}
}

static const __m128i REVERSE = _mm_setr_epi8(-1,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);

static const __m128i MASK7 = {
	14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};

struct HProcessor {

	size_t len;

	size_t vcnt;
	const __m128i* src;
	__m128i* pSumLine;
	__m128i* pSubLine;
	__m128i* pAddLine;
	
	__forceinline
	void calcShiftSum(__m128i& prevSum, __m128i& plus, __m128i& minus, __m128i& sum0, __m128i& sum1)
	{
		__m128i plus0 = _mm_unpacklo_epi8(plus, _mm_setzero_si128());
		__m128i minus0 = _mm_unpacklo_epi8(minus, _mm_setzero_si128());
		__m128i diff0 = _mm_sub_epi16(plus0, minus0);

		__m128i plus1 = _mm_unpackhi_epi8(plus, _mm_setzero_si128());
		__m128i minus1 = _mm_unpackhi_epi8(minus, _mm_setzero_si128());
		__m128i diff1 = _mm_sub_epi16(plus1, minus1);
		
		sum0 = _mm_shuffle_epi8(prevSum, MASK7);
		diff0 = shiftAdd16(diff0);
		sum0 = _mm_add_epi16(sum0, diff0);
		sum1 = _mm_shuffle_epi8(sum0, MASK7);
		diff1 = shiftAdd16(diff1);
		sum1 = _mm_add_epi16(sum1, diff1);
	}
	
	
	template <typename T>
	__forceinline
	void process(T& storer) {

		// îΩì]ÇµÇƒóvëfÇê∂ê¨
		__m128i minusSrc2[2];
		minusSrc2[0] = _mm_shuffle_epi8(src[0], REVERSE);	// 15-1
		minusSrc2[1] = src[0];
		uint16_t sum = 0;
		for (size_t i=0; i<1+len*2; ++i) {
			sum += ((const uint8_t*)(minusSrc2+1)) [i - (len+1)];
		}
		__m128i sum3 = _mm_set1_epi16(sum);
		const __m128i* plusSrc = (const __m128i*) ((const uint8_t*)src + len);
		const __m128i* minusSrc = (const __m128i*) ((const uint8_t*)src - (len+1));
		__m128i plus = _mm_loadu_si128(plusSrc);
		__m128i minus = _mm_loadu_si128(
			(const __m128i*) (
				(const uint8_t*)(minusSrc2+1) - (len+1)
			)
		);
		for (size_t i=0; i<vcnt; ++i) {
			__m128i nextPlus = _mm_loadu_si128(plusSrc+i*2+1);
			__m128i nextMinus = _mm_loadu_si128(minusSrc+i*2+1);
			
			__m128i vsum0 = pSumLine[i*4+0];
			__m128i vsum1 = pSumLine[i*4+1];
			__m128i vsum2 = pSumLine[i*4+2];
			__m128i vsum3 = pSumLine[i*4+3];

			__m128i vminus0 = pSubLine[i*4+0];
			__m128i vminus1 = pSubLine[i*4+1];
			__m128i vminus2 = pSubLine[i*4+2];
			__m128i vminus3 = pSubLine[i*4+3];
			
			__m128i sum0, sum1;
			calcShiftSum(sum3, plus, minus, sum0, sum1);
			
			plus = nextPlus;
			minus = nextMinus;
			
			nextPlus = _mm_loadu_si128(plusSrc+i*2+2);
			nextMinus = _mm_loadu_si128(minusSrc+i*2+2);
			
			__m128i sum2;
			calcShiftSum(sum1, plus, minus, sum2, sum3);

			storer(vsum0, vsum1, vsum2, vsum3);

			vsum0 = _mm_sub_epi16(vsum0, vminus0);
			vsum1 = _mm_sub_epi16(vsum1, vminus1);
			vsum2 = _mm_sub_epi16(vsum2, vminus2);
			vsum3 = _mm_sub_epi16(vsum3, vminus3);
			vsum0 = _mm_add_epi16(vsum0, sum0);
			vsum1 = _mm_add_epi16(vsum1, sum1);
			vsum2 = _mm_add_epi16(vsum2, sum2);
			vsum3 = _mm_add_epi16(vsum3, sum3);

			pAddLine[i*4+0] = sum0;
			pAddLine[i*4+1] = sum1;
			pAddLine[i*4+2] = sum2;
			pAddLine[i*4+3] = sum3;

			pSumLine[i*4+0] = vsum0;
			pSumLine[i*4+1] = vsum1;
			pSumLine[i*4+2] = vsum2;
			pSumLine[i*4+3] = vsum3;
			
			plus = nextPlus;
			minus = nextMinus;
		}
	}

	__forceinline
	void process2() {
		// îΩì]ÇµÇƒóvëfÇê∂ê¨
		__m128i minusSrc2[2];
		minusSrc2[0] = _mm_shuffle_epi8(src[0], REVERSE);	// 15-1
		minusSrc2[1] = src[0];
		uint16_t sum = 0;
		for (size_t i=0; i<1+len*2; ++i) {
			sum += ((const uint8_t*)(minusSrc2+1)) [i - (len+1)];
		}
		__m128i sum3 = _mm_set1_epi16(sum);
		const __m128i* plusSrc = (const __m128i*) ((const uint8_t*)src + len);
		const __m128i* minusSrc = (const __m128i*) ((const uint8_t*)src - (len+1));
		__m128i plus = _mm_loadu_si128(plusSrc);
		__m128i minus = _mm_loadu_si128(
			(const __m128i*) (
				(const uint8_t*)(minusSrc2+1) - (len+1)
			)
		);
		for (size_t i=0; i<vcnt; ++i) {
			__m128i nextPlus = _mm_loadu_si128(plusSrc+i*2+1);
			__m128i nextMinus = _mm_loadu_si128(minusSrc+i*2+1);
			
			__m128i vsum0 = pSumLine[i*4+0];
			__m128i vsum1 = pSumLine[i*4+1];
			__m128i vsum2 = pSumLine[i*4+2];
			__m128i vsum3 = pSumLine[i*4+3];

			__m128i sum0, sum1;
			calcShiftSum(sum3, plus, minus, sum0, sum1);
			
			plus = nextPlus;
			minus = nextMinus;
			
			nextPlus = _mm_loadu_si128(plusSrc+i*2+2);
			nextMinus = _mm_loadu_si128(minusSrc+i*2+2);
			
			__m128i sum2;
			calcShiftSum(sum1, plus, minus, sum2, sum3);

			vsum0 = _mm_add_epi16(vsum0, sum0);
			vsum1 = _mm_add_epi16(vsum1, sum1);
			vsum2 = _mm_add_epi16(vsum2, sum2);
			vsum3 = _mm_add_epi16(vsum3, sum3);
			vsum0 = _mm_add_epi16(vsum0, sum0);
			vsum1 = _mm_add_epi16(vsum1, sum1);
			vsum2 = _mm_add_epi16(vsum2, sum2);
			vsum3 = _mm_add_epi16(vsum3, sum3);

			pAddLine[i*4+0] = sum0;
			pAddLine[i*4+1] = sum1;
			pAddLine[i*4+2] = sum2;
			pAddLine[i*4+3] = sum3;

			pSumLine[i*4+0] = vsum0;
			pSumLine[i*4+1] = vsum1;
			pSumLine[i*4+2] = vsum2;
			pSumLine[i*4+3] = vsum3;

			plus = nextPlus;
			minus = nextMinus;
		}
	}

	__forceinline
	void process3() {
		// îΩì]ÇµÇƒóvëfÇê∂ê¨
		__m128i minusSrc2[2];
		minusSrc2[0] = _mm_shuffle_epi8(src[0], REVERSE);	// 15-1
		minusSrc2[1] = src[0];
		uint16_t sum = 0;
		for (size_t i=0; i<1+len*2; ++i) {
			sum += ((const uint8_t*)(minusSrc2+1)) [i - (len+1)];
		}
		__m128i sum3 = _mm_set1_epi16(sum);
		const __m128i* plusSrc = (const __m128i*) ((const uint8_t*)src + len);
		const __m128i* minusSrc = (const __m128i*) ((const uint8_t*)src - (len+1));
		__m128i plus = _mm_loadu_si128(plusSrc);
		__m128i minus = _mm_loadu_si128(
			(const __m128i*) (
				(const uint8_t*)(minusSrc2+1) - (len+1)
			)
		);
		for (size_t i=0; i<vcnt; ++i) {
			__m128i nextPlus = _mm_loadu_si128(plusSrc+i*2+1);
			__m128i nextMinus = _mm_loadu_si128(minusSrc+i*2+1);
			
			__m128i sum0, sum1;
			calcShiftSum(sum3, plus, minus, sum0, sum1);
			
			plus = nextPlus;
			minus = nextMinus;
			
			nextPlus = _mm_loadu_si128(plusSrc+i*2+2);
			nextMinus = _mm_loadu_si128(minusSrc+i*2+2);
			
			__m128i sum2;
			calcShiftSum(sum1, plus, minus, sum2, sum3);

			pAddLine[i*4+0] = sum0;
			pAddLine[i*4+1] = sum1;
			pAddLine[i*4+2] = sum2;
			pAddLine[i*4+3] = sum3;

			pSumLine[i*4+0] = sum0;
			pSumLine[i*4+1] = sum1;
			pSumLine[i*4+2] = sum2;
			pSumLine[i*4+3] = sum3;

			plus = nextPlus;
			minus = nextMinus;
		}
	}
};

void test_16(const Parameter& p) {

	size_t len = p.radius;
	size_t width = p.width;

	// åªç›ÇÃé¿ëïÇ…ÇÕêßå¿óLÇË
	if (len > 14 || len == 0) {
		return;
	}
	size_t vcnt = width / 32;
	if (vcnt == 0) {
		return;
	}

	const __m128i* src = (const __m128i*) p.pSrc;
	__m128i* dst = (__m128i*) p.pDest;

	__m128i* pSumLine = (__m128i*)p.pWork2;
	typedef RingLinePtr<__m128i*> RingLinePtr128;
	RingLinePtr128 pTopLine(len*2+1, 0, (__m128i*)p.pWork, p.srcLineOffsetBytes*2);
	RingLinePtr128 pAddLine(pTopLine);
	RingLinePtr128 pSubLine(pTopLine);

	HProcessor hProc;
	hProc.len = p.radius;
	hProc.vcnt = vcnt;
	hProc.pSumLine = pSumLine;
	memset(pSumLine, 0, p.srcLineOffsetBytes*2);
	hProc.pSubLine = pSubLine;
	hProc.pAddLine = pAddLine;

	struct Storer
	{
		__m128i invLen;
		__m128i* dst;

		__forceinline
		void operator () (__m128i& vsum0, __m128i& vsum1, __m128i& vsum2, __m128i& vsum3)
		{
			__m128i store0 = _mm_packus_epi16(_mm_mulhi_epu16(vsum0, invLen), _mm_mulhi_epu16(vsum1, invLen));
			__m128i store1 = _mm_packus_epi16(_mm_mulhi_epu16(vsum2, invLen), _mm_mulhi_epu16(vsum3, invLen));
			_mm_stream_si128(dst+0, store0);
			_mm_stream_si128(dst+1, store1);
			dst += 2;
		}
	} storer;
	const size_t diameter = 1 + len * 2;
	storer.invLen = _mm_set1_epi16(0xFFFF / (diameter*diameter));

	hProc.src = src;
	// collect sum (0 to len)
	hProc.process3();
	OffsetPtr(hProc.src, p.srcLineOffsetBytes);
	pAddLine.moveNext();
	hProc.pAddLine = pAddLine;
	for (size_t i=0; i<len; ++i) {
		hProc.process2();
		pAddLine.moveNext();
		OffsetPtr(hProc.src, p.srcLineOffsetBytes);
		hProc.pAddLine = pAddLine;
	}
	// first set (reuse collected data)
	pSubLine = pAddLine;
	pSubLine.movePrev();
	hProc.pSubLine = pSubLine;
	storer.dst = dst;
	for (size_t i=0; i<len; ++i) {
		hProc.process(storer);
		pAddLine.moveNext();
		pSubLine.movePrev();
		hProc.pAddLine = pAddLine;
		hProc.pSubLine = pSubLine;
		OffsetPtr(hProc.src, p.srcLineOffsetBytes);
		OffsetPtr(dst, p.destLineOffsetBytes);
		storer.dst = dst;
	}
	// main set
	for (size_t y=diameter; y<p.height; ++y) {
		hProc.process(storer);
		pAddLine.moveNext();
		pSubLine.moveNext();
		hProc.pSubLine = pSubLine;
		hProc.pAddLine = pAddLine;
		OffsetPtr(hProc.src, p.srcLineOffsetBytes);
		OffsetPtr(dst, p.destLineOffsetBytes);
		storer.dst = dst;
	}

}

void test_20(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	const size_t r = std::min<size_t>(height, std::min<size_t>(width, radius));
	
	const size_t len = r * 2 + 1; // diameter
	size_t invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	struct HorizontalProcessor_Tent {
		const uint16_t width;
		const int r;
		const size_t invCnt;
		
		const size_t kxs0;
		const size_t kxe0;
		
		HorizontalProcessor_Tent(
			uint16_t width,
			int r,
			size_t invCnt
		)
			:
			width(width),
			r(r),
			invCnt(invCnt),
			kxs0( std::min<size_t>(width, 1 + r) ),
			kxe0( std::max<int>(0, width - r) )
		{
		}

		__forceinline void process(
			const uint8_t* __restrict pFrom,
			uint8_t* __restrict pTo
		) {
			const uint8_t* pFromInit = pFrom;
			
			int modi = 0;
			int total = 0;
			const uint8_t* pMinusPlus = pFrom + 1;
			{
				int v = *pFrom++;
				modi = -v;
				total = v * (r + 1);
			}
			for (size_t i=0; i<r; ++i) {
				int v = *pFrom++;
				total += v * (r - i) * 2;
			}
			*pTo++ = (total * invCnt) >> SHIFT;
			const uint8_t* pPlus = pFrom;
			const uint8_t* pMinus = pFrom - 1;
			for (size_t i=0; i<r; ++i) {
				modi += *pPlus++;
				total += modi;
				*pTo++ = (total * invCnt) >> SHIFT;
				int v = *pMinusPlus++;
				modi += - 2 * v + *pMinus--;
			}
			assert(pMinus == pFromInit);
			assert(pPlus == pFromInit + r*2+1);
			const size_t loopCount = width - r * 2 - 1;
			for (size_t i=0; i<loopCount; ++i) {
				modi += *pPlus++;
				total += modi;
				*pTo++ = (total * invCnt) >> SHIFT;
				int v = *pMinusPlus++;
				modi += - 2 * v + *pMinus++;
			}
			pPlus -= 2;
			for (size_t i=0; i<r; ++i) {
				modi += *pPlus--;
				total += modi;
				*pTo++ = (total * invCnt) >> SHIFT;
				int v = *pMinusPlus++;
				modi += - 2 * v + *pMinus++;
			}
			
		}
	};

	struct VerticalProcessor_Tint {
		const size_t width;
		const size_t invCnt;
		int16_t* pModiLine;
		uint32_t* pTotalLine;
		
		VerticalProcessor_Tint(
			size_t width,
			size_t invCnt,
			int16_t* pModiLine,
			uint32_t* pTotalLine
		)
			:
			width(width),
			invCnt(invCnt),
			pModiLine(pModiLine),
			pTotalLine(pTotalLine)
		{
		}
		void process(
			const uint8_t* pMinusSrcLine,
			const uint8_t* pMinusPlusSrcLine,
			const uint8_t* pPlusSrcLine,
			uint8_t* pToLine
		) {
			for (size_t x=0; x<width; ++x) {
				int modi = pModiLine[x];
				modi += pPlusSrcLine[x];
				size_t total = pTotalLine[x] + modi;
				pToLine[x] = (total * invCnt) >> SHIFT;
				int v = pMinusPlusSrcLine[x];
				modi += -2 * v + pMinusSrcLine[x];
				pModiLine[x] = modi;
				pTotalLine[x] = total;
			}	
		}
	};

	
	const size_t invCnt = (1<<SHIFT) / ((r + 1) * (r + 1));
	HorizontalProcessor_Tent horizontal(width, r, invCnt);
	int16_t* pModiLine = (int16_t*)p.pModi;
	uint32_t* pTotalLine = (uint32_t*)p.pTotal;
	VerticalProcessor_Tint vertical(width, invCnt, pModiLine, pTotalLine);
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFromLine;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFromLine = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			pFromLine = pWork2;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pToLine;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pToLine = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pToLine = pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		RingLinePtr<uint8_t*> pWorkLine(len+1, 0, pWork, workLineOffsetBytes);
		RingLinePtr<uint8_t*> pMinusPlusSrcLine = pWorkLine;
		RingLinePtr<uint8_t*> pPlusSrcLine = pWorkLine;
		RingLinePtr<uint8_t*> pMinusSrcLine = pWorkLine;
		pMinusPlusSrcLine.moveNext();
		if (bTop) {
			horizontal.process(pFromLine, pWorkLine);
			for (size_t x=0; x<width; ++x) {
				int v = pWorkLine[x];
				pModiLine[x] = -v;
				pTotalLine[x] = v * (r + 1);
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pWorkLine.moveNext();
			for (size_t i=0; i<r; ++i) {
				horizontal.process(pFromLine, pWorkLine);
				const size_t factor = (r - i) * 2;
				for (size_t x=0; x<width; ++x) {
					int v = pWorkLine[x];
					pTotalLine[x] += v * factor;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invCnt) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
			pPlusSrcLine = pWorkLine;
			pMinusSrcLine = pWorkLine;
			pMinusSrcLine.movePrev();
			for (size_t i=0; i<r; ++i) {
				horizontal.process(pFromLine, pPlusSrcLine);
				vertical.process(pMinusSrcLine, pMinusPlusSrcLine, pPlusSrcLine, pToLine);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pPlusSrcLine.moveNext();
				pMinusPlusSrcLine.moveNext();
				pMinusSrcLine.movePrev();
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}else {
			OffsetPtr(pFromLine, -r * fromLineOffsetBytes);
			horizontal.process(pFromLine, pWorkLine);
			pMinusSrcLine = pWorkLine;
			for (size_t x=0; x<width; ++x) {
				int v = pWorkLine[x];
				pTotalLine[x] = v;
				pModiLine[x] = -v;
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pWorkLine.moveNext();
			for (size_t i=0; i<r; ++i) {
				horizontal.process(pFromLine, pWorkLine);
				const size_t factor = i + 2;
				for (size_t x=0; x<width; ++x) {
					int v = pWorkLine[x];
					pTotalLine[x] += v * factor;
					pModiLine[x] -= v;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			pMinusPlusSrcLine = pWorkLine;
			for (size_t i=0; i<r; ++i) {
				horizontal.process(pFromLine, pWorkLine);
				const size_t factor = r - i;
				for (size_t x=0; x<width; ++x) {
					int v = pWorkLine[x];
					pTotalLine[x] += v * factor;
					pModiLine[x] += v;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			pPlusSrcLine = pWorkLine;
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invCnt) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		size_t loopCount = height - 1;
		if (bTop) {
			loopCount -= r;
		}else {
			loopCount += (iterationCount - n - 1) * r;
		}
		if (bBottom) {
			loopCount -= r;
		}else {
			loopCount += (iterationCount - n - 1) * r;
		}
		for (size_t i=0; i<loopCount; ++i) {
			horizontal.process(pFromLine, pPlusSrcLine);
			vertical.process(pMinusSrcLine, pMinusPlusSrcLine, pPlusSrcLine, pToLine);
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pPlusSrcLine.moveNext();
			pMinusPlusSrcLine.moveNext();
			pMinusSrcLine.moveNext();
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		if (bBottom) {
			pPlusSrcLine.move(-2);
			for (size_t i=0; i<r; ++i) {
				vertical.process(pMinusSrcLine, pMinusPlusSrcLine, pPlusSrcLine, pToLine);
				pPlusSrcLine.movePrev();
				pMinusPlusSrcLine.moveNext();
				pMinusSrcLine.moveNext();
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}
	}
	
}

struct HorizontalProcessor_Tent {
	const uint16_t width;
	const int r;
	const size_t invCnt;
	
	const size_t kxs0;
	const size_t kxe0;
	
	HorizontalProcessor_Tent(
		uint16_t width,
		int r,
		size_t invCnt
	)
		:
		width(width),
		r(r),
		invCnt(invCnt),
		kxs0( std::min<size_t>(width, 1 + r) ),
		kxe0( std::max<int>(0, width - r) )
	{
	}

	__forceinline void process(
		const uint8_t* __restrict pFrom,
		uint8_t* __restrict pTo
	) {
		const uint8_t* pFromInit = pFrom;
		
		int modi = 0;
		int total = 0;
		const uint8_t* pMinusPlus = pFrom + 1;
		{
			int v = *pFrom++;
			modi = -v;
			total = v * (r + 1);
		}
		for (size_t i=0; i<r; ++i) {
			int v = *pFrom++;
			total += v * (r - i) * 2;
		}
		*pTo++ = (total * invCnt) >> SHIFT;
		const uint8_t* pPlus = pFrom;
		const uint8_t* pMinus = pFrom - 1;
		for (size_t i=0; i<r; ++i) {
			modi += *pPlus++;
			total += modi;
			*pTo++ = (total * invCnt) >> SHIFT;
			int v = *pMinusPlus++;
			modi += - 2 * v + *pMinus--;
		}
		assert(pMinus == pFromInit);
		assert(pPlus == pFromInit + r*2+1);
		const size_t loopCount = width - r * 2 - 1;
		for (size_t i=0; i<loopCount; ++i) {
			modi += *pPlus++;
			total += modi;
			*pTo++ = (total * invCnt) >> SHIFT;
			int v = *pMinusPlus++;
			modi += - 2 * v + *pMinus++;
		}
		pPlus -= 2;
		for (size_t i=0; i<r; ++i) {
			modi += *pPlus--;
			total += modi;
			*pTo++ = (total * invCnt) >> SHIFT;
			int v = *pMinusPlus++;
			modi += - 2 * v + *pMinus++;
		}
		
	}
};

struct VerticalProcessor_Tint {
	const size_t width;
	const size_t invCnt;
	const __m128i mInvCnt;
	__m128i* pModiLine;
	__m128i* pTotalLine;
	
	VerticalProcessor_Tint(
		size_t width,
		size_t invCnt,
		__m128i* pModiLine,
		__m128i* pTotalLine
	)
		:
		width(width),
		invCnt(invCnt),
		mInvCnt(_mm_set1_epi16(invCnt)),
		pModiLine(pModiLine),
		pTotalLine(pTotalLine)
	{
	}
	__forceinline void process(
		const __m128i* pMinusSrcLine,
		const __m128i* pMinusPlusSrcLine,
		const __m128i* pPlusSrcLine,
		__m128i* pToLine
	) {
		const size_t loopCount = width / 16;
		for (size_t x=0; x<loopCount; ++x) {
			__m128i plus = pPlusSrcLine[x];
			__m128i plus0 = _mm_unpacklo_epi8(plus, _mm_setzero_si128());
			__m128i plus1 = _mm_unpackhi_epi8(plus, _mm_setzero_si128());
			__m128i modi0 = _mm_add_epi16(pModiLine[x*2+0], plus0);
			__m128i modi1 = _mm_add_epi16(pModiLine[x*2+1], plus1);
			__m128i modi0Sign = _mm_cmplt_epi16(modi0, _mm_setzero_si128());
			__m128i modi1Sign = _mm_cmplt_epi16(modi1, _mm_setzero_si128());
			
			__m128i totalA = pTotalLine[x*4+0];
			__m128i totalB = pTotalLine[x*4+1];
			totalA = _mm_add_epi32(totalA, _mm_unpacklo_epi16(modi0, modi0Sign));
			totalB = _mm_add_epi32(totalB, _mm_unpackhi_epi16(modi0, modi0Sign));
			pTotalLine[x*4+0] = totalA;
			pTotalLine[x*4+1] = totalB;
			__m128i total0 = _mm_packs_epi32(totalA, totalB);
			
			__m128i totalC = pTotalLine[x*4+2];
			__m128i totalD = pTotalLine[x*4+3];
			
			totalC = _mm_add_epi32(totalC, _mm_unpacklo_epi16(modi1, modi1Sign));
			totalD = _mm_add_epi32(totalD, _mm_unpackhi_epi16(modi1, modi1Sign));
			pTotalLine[x*4+2] = totalC;
			pTotalLine[x*4+3] = totalD;
			__m128i total1 = _mm_packs_epi32(totalC, totalD);
			
			__m128i result0 = _mm_mulhi_epu16(total0, mInvCnt);
			__m128i result1 = _mm_mulhi_epu16(total1, mInvCnt);
			__m128i result = _mm_packus_epi16(result0, result1);
			
//			_mm_stream_si128(pToLine+x, result);
			pToLine[x] = result;
			
			__m128i minusPlus = pMinusPlusSrcLine[x];
			__m128i minusPlus0 = _mm_unpacklo_epi8(minusPlus, _mm_setzero_si128());
			__m128i minusPlus1 = _mm_unpackhi_epi8(minusPlus, _mm_setzero_si128());
			__m128i minus = pMinusSrcLine[x];
			__m128i minus0 = _mm_unpacklo_epi8(minus, _mm_setzero_si128());
			__m128i minus1 = _mm_unpackhi_epi8(minus, _mm_setzero_si128());
			modi0 = _mm_add_epi16(modi0, minus0);
			modi0 = _mm_sub_epi16(modi0, minusPlus0);
			modi0 = _mm_sub_epi16(modi0, minusPlus0);
			modi1 = _mm_add_epi16(modi1, minus1);
			modi1 = _mm_sub_epi16(modi1, minusPlus1);
			modi1 = _mm_sub_epi16(modi1, minusPlus1);
			
			pModiLine[x*2+0] = modi0;
			pModiLine[x*2+1] = modi1;
		}
		const size_t remainCount = width & 0x0f;
		int16_t* pModiLine2 = (int16_t*) pModiLine;
		const uint8_t* pPlusSrcLine2 = (uint8_t*) pPlusSrcLine;
		const uint8_t* pMinusPlusSrcLine2 = (uint8_t*) pMinusPlusSrcLine;
		const uint8_t* pMinusSrcLine2 = (uint8_t*) pMinusSrcLine;
		uint32_t* pTotalLine2 = (uint32_t*) pTotalLine;
		uint8_t* pToLine2 = (uint8_t*) pToLine;
		for (size_t x=0; x<remainCount; ++x) {
			int modi = pModiLine2[x];
			modi += pPlusSrcLine2[x];
			size_t total = pTotalLine2[x] + modi;
			pToLine2[x] = (total * invCnt) >> SHIFT;
			int v = pMinusPlusSrcLine2[x];
			modi += -2 * v + pMinusSrcLine2[x];
			pModiLine2[x] = modi;
			pTotalLine2[x] = total;
		}
	}
};

void test_21(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	const size_t r = std::min<size_t>(height, std::min<size_t>(width, radius));
	
	const size_t len = r * 2 + 1; // diameter
	size_t invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}
	
	const size_t invCnt = (1<<SHIFT) / ((r + 1) * (r + 1));
	HorizontalProcessor_Tent horizontal(width, r, invCnt);
	int16_t* pModiLine = (int16_t*)p.pModi;
	uint32_t* pTotalLine = (uint32_t*)p.pTotal;
	VerticalProcessor_Tint vertical(width, invCnt, (__m128i*)pModiLine, (__m128i*)pTotalLine);
	
	for (size_t n=0; n<iterationCount; ++n) {
		
		const uint8_t* pFromLine;
		ptrdiff_t fromLineOffsetBytes;
		if (n == 0) {
			pFromLine = pSrc;
			fromLineOffsetBytes = srcLineOffsetBytes;
		}else {
			pFromLine = pWork2;
			fromLineOffsetBytes = workLineOffsetBytes;
		}
		uint8_t* pToLine;
		ptrdiff_t toLineOffsetBytes;
		if (n == iterationCount - 1) {
			pToLine = pDest;
			toLineOffsetBytes = destLineOffsetBytes;
		}else {
			pToLine = pWork2;
			toLineOffsetBytes = workLineOffsetBytes;
		}
		
		RingLinePtr<uint8_t*> pWorkLine(len+1, 0, pWork, workLineOffsetBytes);
		RingLinePtr<uint8_t*> pMinusPlusSrcLine = pWorkLine;
		RingLinePtr<uint8_t*> pPlusSrcLine = pWorkLine;
		RingLinePtr<uint8_t*> pMinusSrcLine = pWorkLine;
		pMinusPlusSrcLine.moveNext();
		if (bTop) {
			horizontal.process(pFromLine, pWorkLine);
			for (size_t x=0; x<width; ++x) {
				int v = pWorkLine[x];
				pModiLine[x] = -v;
				pTotalLine[x] = v * (r + 1);
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pWorkLine.moveNext();
			for (size_t i=0; i<r; ++i) {
				horizontal.process(pFromLine, pWorkLine);
				const size_t factor = (r - i) * 2;
				for (size_t x=0; x<width; ++x) {
					int v = pWorkLine[x];
					pTotalLine[x] += v * factor;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invCnt) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
			pPlusSrcLine = pWorkLine;
			pMinusSrcLine = pWorkLine;
			pMinusSrcLine.movePrev();
			for (size_t i=0; i<r; ++i) {
				horizontal.process(pFromLine, pPlusSrcLine);
				vertical.process((const __m128i*)pMinusSrcLine, (const __m128i*)pMinusPlusSrcLine, (const __m128i*)pPlusSrcLine, (__m128i*)pToLine);
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pPlusSrcLine.moveNext();
				pMinusPlusSrcLine.moveNext();
				pMinusSrcLine.movePrev();
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}else {
			OffsetPtr(pFromLine, -r * fromLineOffsetBytes);
			horizontal.process(pFromLine, pWorkLine);
			pMinusSrcLine = pWorkLine;
			for (size_t x=0; x<width; ++x) {
				int v = pWorkLine[x];
				pTotalLine[x] = v;
				pModiLine[x] = -v;
			}
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pWorkLine.moveNext();
			for (size_t i=0; i<r; ++i) {
				horizontal.process(pFromLine, pWorkLine);
				const size_t factor = i + 2;
				for (size_t x=0; x<width; ++x) {
					int v = pWorkLine[x];
					pTotalLine[x] += v * factor;
					pModiLine[x] -= v;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			pMinusPlusSrcLine = pWorkLine;
			for (size_t i=0; i<r; ++i) {
				horizontal.process(pFromLine, pWorkLine);
				const size_t factor = r - i;
				for (size_t x=0; x<width; ++x) {
					int v = pWorkLine[x];
					pTotalLine[x] += v * factor;
					pModiLine[x] += v;
				}
				OffsetPtr(pFromLine, fromLineOffsetBytes);
				pWorkLine.moveNext();
			}
			pPlusSrcLine = pWorkLine;
			for (size_t x=0; x<width; ++x) {
				pToLine[x] = (pTotalLine[x] * invCnt) >> SHIFT;
			}
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		size_t loopCount = height - 1;
		if (bTop) {
			loopCount -= r;
		}else {
			loopCount += (iterationCount - n - 1) * r;
		}
		if (bBottom) {
			loopCount -= r;
		}else {
			loopCount += (iterationCount - n - 1) * r;
		}
		for (size_t i=0; i<loopCount; ++i) {
			horizontal.process(pFromLine, pPlusSrcLine);
			vertical.process((const __m128i*)pMinusSrcLine, (const __m128i*)pMinusPlusSrcLine, (const __m128i*)pPlusSrcLine, (__m128i*)pToLine);
			OffsetPtr(pFromLine, fromLineOffsetBytes);
			pPlusSrcLine.moveNext();
			pMinusPlusSrcLine.moveNext();
			pMinusSrcLine.moveNext();
			OffsetPtr(pToLine, toLineOffsetBytes);
		}
		
		if (bBottom) {
			pPlusSrcLine.move(-2);
			for (size_t i=0; i<r; ++i) {
				vertical.process((const __m128i*)pMinusSrcLine, (const __m128i*)pMinusPlusSrcLine, (const __m128i*)pPlusSrcLine, (__m128i*)pToLine);
				pPlusSrcLine.movePrev();
				pMinusPlusSrcLine.moveNext();
				pMinusSrcLine.moveNext();
				OffsetPtr(pToLine, toLineOffsetBytes);
			}
		}
	}
	
}

} // namespace blur_1b
