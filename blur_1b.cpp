
#include "blur_1b.h"
#include <algorithm>
#include <smmintrin.h>

namespace blur_1b {

static const int SHIFT = 15;

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
	const ptrdiff_t workLineOffsetBytes = p.workLineOffsetBytes;\
	int16_t* pTotalLine = p.pTotalLine;\
	const uint8_t radius = p.radius;

template <typename T>
class Image {
private:
	T* mPtr;
	const size_t mWidth;
	const size_t mHeight;
	const ptrdiff_t mLineOffsetBytes;
public:
	Image(T* ptr, size_t width, size_t height, ptrdiff_t lineOffsetBytes)
		:
		mPtr(ptr),
		mWidth(width),
		mHeight(height),
		mLineOffsetBytes(lineOffsetBytes)
	{
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


void test_1(const Parameter& p) {

	BLUR_EXTRACT_PARAMS;
	
	Image<const uint8_t> src(pSrc, width, height, srcLineOffsetBytes);
	Image<uint8_t> target(pDest, width, height, destLineOffsetBytes);
	const int r = radius;
	double len = r * 2 + 1;
	double area = len * len;
	double invArea = 1.0 / area;
	for (size_t y=0; y<height; ++y) {
		for (size_t x=0; x<width; ++x) {
			unsigned int total = 0;
			for (int ky=-r; ky<=r; ++ky) {
				for (int kx=-r; kx<=r; ++kx) {
					total += src.get(x+kx, y+ky);
				}
			}
			target.set(x, y, (uint8_t)(total * invArea + 0.5));
		}
	}
}

void test_2(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int r = p.radius;
	double len = r * 2 + 1;
	double area = len * len;
	double invArea = 1.0 / area;
	
	uint8_t* pDestLine = pDest;
	
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
			const uint8_t* pSrcLine = pSrc;
			OffsetPtr(pSrcLine, kys*srcLineOffsetBytes);
			for (int ky=kys; ky<=kye; ++ky) {
				for (int kx=kxs; kx<=kxe; ++kx) {
					total += pSrcLine[kx];
				}
				OffsetPtr(pSrcLine, srcLineOffsetBytes);
			}
			pDestLine[x] = (uint8_t)(total * invArea + 0.5);
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
}

void test_3(const Parameter& p) {

	BLUR_EXTRACT_PARAMS;
	
	int r = radius;
	double len = r * 2 + 1;
	double area = len * len;
	double invLen = 1.0 / len;
	double invArea = 1.0 / area;

	// horizontal
	{
		uint8_t* pWorkLine = pWork;
		const uint8_t* pSrcLine = pSrc;
		for (size_t y=0; y<height; ++y) {
			for (size_t x=0; x<width; ++x) {
				unsigned int total = 0;
				const int kxs = std::max<int>(0, x-r);
				const int kxe = std::min<int>(width, x+r);
				for (int kx=kxs; kx<=kxe; ++kx) {
					total += pSrcLine[kx];
				}
				pWorkLine[x] = (uint8_t)(total * invLen + 0.5);
			}
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
		}
	}
	// vertical
	{
		const uint8_t* pWorkLine = pWork;
		uint8_t* pDestLine = pDest;
		for (size_t x=0; x<width; ++x) {
			pDestLine = pDest;
			pDestLine += x;
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
				*pDestLine = (uint8_t)(total * invLen + 0.5);
				OffsetPtr(pDestLine, destLineOffsetBytes);
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

	// horizontal
	{
		uint8_t* pWorkLine = pWork;
		const uint8_t* pSrcLine = pSrc;
		for (size_t y=0; y<height; ++y) {
			for (size_t x=0; x<width; ++x) {
				unsigned int total = 0;
				const int kxs = std::max<int>(0, x-r);
				const int kxe = std::min<int>(width, x+r);
				for (int kx=kxs; kx<=kxe; ++kx) {
					total += pSrcLine[kx];
				}
				pWorkLine[x] = (total * invLen) >> SHIFT;
			}
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
		}
	}
	// vertical
	{
		const uint8_t* pWorkLine = pWork;
		uint8_t* pDestLine = pDest;
		for (size_t x=0; x<width; ++x) {
			pDestLine = pDest + x;
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
				*pDestLine = (total * invLen) >> SHIFT;
				OffsetPtr(pDestLine, destLineOffsetBytes);
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
	const uint8_t* pSrcLine = pSrc;
	uint8_t* pWorkLine = pWork;
	for (size_t y=0; y<height; ++y) {
		int total = *pSrcLine;
		for (size_t kx=1; kx<kxs0; ++kx) {
			total += pSrcLine[kx] * 2;
		}
		pWorkLine[0] = (total * invLen) >> SHIFT;
		for (size_t x=1; x<kxs0; ++x) {
			assert(kxs0 >= x);
			total -= pSrcLine[kxs0 - x];
			total += pSrcLine[kxs0 + x - 1];
			pWorkLine[x] = (total * invLen) >> SHIFT;
		}
		for (size_t x=kxs0; x<kxe0; ++x) {
			total -= pSrcLine[x - r - 1];
			total += pSrcLine[x + r];
			pWorkLine[x] = (total * invLen) >> SHIFT;
		}
		for (size_t x=kxe0,cnt=0; x<width; ++x, ++cnt) {
			total -= pSrcLine[kxe0 - r + cnt];
			total += pSrcLine[width - 1 - cnt];
			pWorkLine[x] = (total * invLen) >> SHIFT;
		}
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		OffsetPtr(pSrcLine, srcLineOffsetBytes);
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
	const uint8_t* pWorkLine = pWork;
	const uint8_t* pWorkLine2 = pWorkLine;
	uint8_t* pDestLine = pDest;
	for (size_t x=0; x<width; ++x) {
		
		pDestLine = pDest + x;
		const uint8_t* pWorkBase = pWork + x;
		
		int total = 0;
		
		if (bTop) {
			pWorkLine = pWorkBase;
			pWorkLine2 = pWorkLine;
			OffsetPtr(pWorkLine2, r * workLineOffsetBytes);
			
			total = *pWorkLine;
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			for (size_t ky=1; ky<=r; ++ky) {
				total += *pWorkLine * 2;
				OffsetPtr(pWorkLine, workLineOffsetBytes);
			}
			
			*pDestLine = (total * invLen) >> SHIFT;
			OffsetPtr(pDestLine, destLineOffsetBytes);
			
			for (size_t y=1; y<=r; ++y) {
				total -= *pWorkLine2;
				total += *pWorkLine;
				OffsetPtr(pWorkLine2, -workLineOffsetBytes);
				OffsetPtr(pWorkLine, workLineOffsetBytes);
				*pDestLine = (total * invLen) >> SHIFT;
				OffsetPtr(pDestLine, destLineOffsetBytes);
			}
		}else {
			pWorkLine = pWorkBase;
			OffsetPtr(pWorkLine, -r * workLineOffsetBytes);
			pWorkLine2 = pWorkLine;
			for (int ky=-r; ky<=r; ++ky) {
				total += *pWorkLine;
				OffsetPtr(pWorkLine, workLineOffsetBytes);
			}
			*pDestLine = (total * invLen) >> SHIFT;
			OffsetPtr(pDestLine, destLineOffsetBytes);

		}

		for (size_t y=kys0; y<kye0; ++y) {
			total -= *pWorkLine2;
			total += *pWorkLine;
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pWorkLine2, workLineOffsetBytes);
			*pDestLine = (total * invLen) >> SHIFT;
			OffsetPtr(pDestLine, destLineOffsetBytes);
		}
		
		if (bBottom) {
			pWorkLine2 = pWorkBase;
			OffsetPtr(pWorkLine2, (kye0 - r) * workLineOffsetBytes);
			pWorkLine = pWorkBase;
			OffsetPtr(pWorkLine, (height - 1) * workLineOffsetBytes);
			for (size_t y=kye0,cnt=0; y<height; ++y, ++cnt) {
				total -= *pWorkLine2;
				total += *pWorkLine;
				OffsetPtr(pWorkLine2, workLineOffsetBytes);
				OffsetPtr(pWorkLine, -workLineOffsetBytes);
				*pDestLine = (total * invLen) >> SHIFT;
				OffsetPtr(pDestLine, destLineOffsetBytes);
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
	
	const uint8_t* pWorkLine = pWork;
	const uint8_t* pWorkLine2 = pWorkLine;
	uint8_t* pDestLine = pDest;
			
	if (bTop) {
		for (size_t x=0; x<width; ++x) {
			pTotalLine[x] = pWorkLine[x];
		}
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		for (size_t ky=1; ky<=r; ++ky) {
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] += pWorkLine[x] * 2;
			}
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
		OffsetPtr(pWorkLine2, r * workLineOffsetBytes);
		
		for (size_t y=1; y<=r; ++y) {
			for (size_t x=0; x<width; ++x) {
				int total = pTotalLine[x] - pWorkLine2[x] + pWorkLine[x];
				pDestLine[x] = (total * invLen) >> SHIFT;
				pTotalLine[x] = total;
			}
			OffsetPtr(pWorkLine2, -workLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
		}
	}else {
		for (size_t x=0; x<width; ++x) {
			pTotalLine[x] = 0;
		}
		OffsetPtr(pWorkLine, -r * workLineOffsetBytes);
		pWorkLine2 = pWorkLine;
		for (int y=-r; y<=r; ++y) {
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] += pWorkLine[x];
			}
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	for (size_t y=kys0; y<kye0; ++y) {
		for (size_t x=0; x<width; ++x) {
			int total = pTotalLine[x] - pWorkLine2[x] + pWorkLine[x];
			pDestLine[x] = (total * invLen) >> SHIFT;
			pTotalLine[x] = total;
		}
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		OffsetPtr(pWorkLine2, workLineOffsetBytes);
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	if (bBottom) {
		pWorkLine2 = pWork;
		OffsetPtr(pWorkLine2, (kye0 - r) * workLineOffsetBytes);
		pWorkLine = pWork;
		OffsetPtr(pWorkLine, (height - 1) * workLineOffsetBytes);
		for (size_t y=kye0,cnt=0; y<height; ++y, ++cnt) {
			for (size_t x=0; x<width; ++x) {
				int total = pTotalLine[x] - pWorkLine2[x] + pWorkLine[x];
				pDestLine[x] = (total * invLen) >> SHIFT;
				pTotalLine[x] = total;
			}
			OffsetPtr(pWorkLine2, workLineOffsetBytes);
			OffsetPtr(pWorkLine, -workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
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
	const uint8_t* pSrcLine = pSrc;
	uint8_t* pWorkLine = pWork;
	for (size_t y=0; y<height; ++y) {
		int total = *pSrcLine;
		for (size_t kx=1; kx<kxs0; ++kx) {
			total += pSrcLine[kx] * 2;
		}
		pWorkLine[0] = (total * invLen) >> SHIFT;
		for (size_t x=1; x<kxs0; ++x) {
			assert(kxs0 >= x);
			total += - pSrcLine[kxs0 - x] + pSrcLine[kxs0 + x - 1];
			pWorkLine[x] = (total * invLen) >> SHIFT;
		}
		for (size_t x=kxs0; x<kxs0_16end; ++x) {
			total += - pSrcLine[x - r - 1] + pSrcLine[x + r];
			pWorkLine[x] = (total * invLen) >> SHIFT;
		}
		
		__m128i* mPSub = (__m128i*) (pSrcLine + kxs0_16end - r - 1);
		__m128i* mPAdd = (__m128i*) (pSrcLine + kxs0_16end + r);
		__m128i mNextSub = _mm_loadu_si128(mPSub++); // hoist loading
		__m128i mNextAdd = _mm_loadu_si128(mPAdd++);
		__m128i* mPWork = (__m128i*) (pWorkLine + kxs0_16end);

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
			__m128i left2 = _mm_mulhrs_epi16(left, mInvLeni);
			__m128i right2 = _mm_mulhrs_epi16(right, mInvLeni);
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
			total += - pSrcLine[x - r - 1] + pSrcLine[x + r];
			pWorkLine[x] = (total * invLen) >> SHIFT;
		}
		for (size_t x=kxe0,cnt=0; x<width; ++x, ++cnt) {
			total += - pSrcLine[kxe0 - r + cnt] + pSrcLine[width - 1 - cnt];
			pWorkLine[x] = (total * invLen) >> SHIFT;
		}
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		OffsetPtr(pSrcLine, srcLineOffsetBytes);
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
				
				__m128i mDest0 = _mm_mulhrs_epi16(mTotal0, mInvLen);
				__m128i mDest1 = _mm_mulhrs_epi16(mTotal1, mInvLen);
				
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
	const uint8_t* pWorkLine = pWork;
	const uint8_t* pWorkLine2 = pWorkLine;
	uint8_t* pDestLine = pDest;
	
	if (bTop) {
		for (size_t x=0; x<width; ++x) {
			pTotalLine[x] = pWorkLine[x];
		}
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		for (size_t ky=1; ky<=r; ++ky) {
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] += pWorkLine[x] * 2;
			}
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
		OffsetPtr(pWorkLine2, r * workLineOffsetBytes);
		
		for (size_t y=1; y<=r; ++y) {
			Worker::process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pDestLine, mInvLeni, width);
			OffsetPtr(pWorkLine2, -workLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
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
		OffsetPtr(pWorkLine, -r * workLineOffsetBytes);
		pWorkLine2 = pWorkLine;
		for (int ky=-r; ky<=r; ++ky) {
			const __m128i* pMWork = (const __m128i*)pWorkLine;
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
				pTotalLine[x] += pWorkLine[x];
			}
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	for (int y=kys0; y<kye0; ++y) {
		Worker::process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pDestLine, mInvLeni, width);
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		OffsetPtr(pWorkLine2, workLineOffsetBytes);
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	if (bBottom) {
		pWorkLine2 = pWork;
		OffsetPtr(pWorkLine2, (kye0 - r) * workLineOffsetBytes);
		pWorkLine = pWork;
		OffsetPtr(pWorkLine, (height - 1) * workLineOffsetBytes);
		for (size_t y=kye0,cnt=0; y<height; ++y, ++cnt) {
			Worker::process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pDestLine, mInvLeni, width);
			OffsetPtr(pWorkLine2, workLineOffsetBytes);
			OffsetPtr(pWorkLine, -workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
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
	
	const uint8_t* pSrcLine = pSrc;
	uint8_t* pWorkLine = pWork;
	uint8_t* pWorkLine2 = pWorkLine;
	uint8_t* pDestLine = pDest;
	
	HorizontalCollector horizontal(width, r);
	
	struct MovOperator {
		uint8_t* __restrict pXDest;
		int16_t* __restrict pYTotal;
		const int invLen;
		
		MovOperator(int invLen) : invLen(invLen) {}
		
		__forceinline void process(int32_t total) {
			int v = (total * invLen) >> SHIFT;
			*pXDest++ = v;
			*pYTotal++ = v;
		}
	};

	struct AddOperator {
		uint8_t* __restrict pXDest;
		int16_t* __restrict pYTotal;
		int invLen;
		
		AddOperator(int invLen) : invLen(invLen) {}

		__forceinline void process(int32_t total) {
			int v = (total * invLen) >> SHIFT;
			*pXDest++ = v;
			*pYTotal++ += v;
		}
	};

	struct Add2Operator {
		uint8_t* __restrict pXDest;
		int16_t* __restrict pYTotal;
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
		int16_t* __restrict pYTotal;
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
	
	SlideOperator slideOp(invLen);
	
	if (bTop) {
		MovOperator movOp(invLen);
		Add2Operator add2Op(invLen);
		movOp.pYTotal = pTotalLine;
		movOp.pXDest = pWorkLine;
		horizontal.process(pSrcLine, movOp);
		OffsetPtr(pSrcLine, srcLineOffsetBytes);
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		for (size_t ky=1; ky<=r; ++ky) {
			add2Op.pXDest = pWorkLine;
			add2Op.pYTotal = pTotalLine;
			horizontal.process(pSrcLine, add2Op);
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
		OffsetPtr(pWorkLine2, r * workLineOffsetBytes);
		
		for (size_t y=1; y<=r; ++y) {
			slideOp.pYTotal = pTotalLine;
			slideOp.pXDest = pWorkLine;
			slideOp.pYSub = pWorkLine2;
			slideOp.pYDest = pDestLine;
			horizontal.process(pSrcLine, slideOp);
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine2, -workLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
		}
	}else {
		AddOperator addOp(invLen);
		for (size_t x=0; x<width; ++x) {
			pTotalLine[x] = 0;
		}
		OffsetPtr(pSrcLine, -r * srcLineOffsetBytes);
		OffsetPtr(pWorkLine, -r * workLineOffsetBytes);
		pWorkLine2 = pWorkLine;
		for (int y=-r; y<=r; ++y) {
			addOp.pYTotal = pTotalLine;
			addOp.pXDest = pWorkLine;
			horizontal.process(pSrcLine, addOp);
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	for (size_t y=kys0; y<kye0; ++y) {
		slideOp.pYTotal = pTotalLine;
		slideOp.pXDest = pWorkLine;
		slideOp.pYSub = pWorkLine2;
		slideOp.pYDest = pDestLine;
		horizontal.process(pSrcLine, slideOp);
		OffsetPtr(pSrcLine, srcLineOffsetBytes);
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		OffsetPtr(pWorkLine2, workLineOffsetBytes);
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	if (bBottom) {
		pWorkLine2 = pWork;
		OffsetPtr(pWorkLine2, (kye0 - r) * workLineOffsetBytes);
		pWorkLine = pWork;
		OffsetPtr(pWorkLine, (height - 1) * workLineOffsetBytes);
		for (size_t y=kye0,cnt=0; y<height; ++y, ++cnt) {
			for (size_t x=0; x<width; ++x) {
				int total = pTotalLine[x] - pWorkLine2[x] + pWorkLine[x];
				pDestLine[x] = (total * invLen) >> SHIFT;
				pTotalLine[x] = total;
			}
			OffsetPtr(pWorkLine2, workLineOffsetBytes);
			OffsetPtr(pWorkLine, -workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
		}
	}
	
}

void test_9(const Parameter& p) {
	
	BLUR_EXTRACT_PARAMS;
	
	int r = std::min<int>(height, std::min<int>(width, radius));
	
	int len = r * 2 + 1; // diameter
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}

	// 横を処理すると同時に縦の処理も行えるので行う。メモリとのやり取りが減るので処理の高速化に繋がる。
	// 他のスレッド含めた横の処理が完了してからでないと縦の領域の処理を行えない場合があるが、それは望ましくない。
	// 横の処理を端を超えて余分に行う事によって待つ必要が無くなる。
	// vertical
	int kys0 = 1;
	int kye0 = height;
	if (bTop) {
		kys0 = std::min<int>(height, 1 + r);
	}
	if (bBottom) {
		kye0 = std::max<int>(0, height - r);
	}
	
	const uint8_t* pSrcLine = pSrc;
	uint8_t* pWorkLine = pWork;
	uint8_t* pWorkLine2 = pWorkLine;
	uint8_t* pDestLine = pDest;

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
		
		__forceinline void process(const uint8_t* __restrict pSrc, uint8_t* __restrict pWork) {
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
	
	if (bTop) {
		horizontal.process(pSrcLine, pWorkLine);
		for (size_t x=0; x<width; ++x) {
			pTotalLine[x] = pWorkLine[x];
		}
		OffsetPtr(pSrcLine, srcLineOffsetBytes);
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		for (size_t ky=1; ky<=r; ++ky) {
			horizontal.process(pSrcLine, pWorkLine);
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] += pWorkLine[x] * 2;
			}
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
		OffsetPtr(pWorkLine2, r * workLineOffsetBytes);
		
		for (size_t y=1; y<=r; ++y) {
			horizontal.process(pSrcLine, pWorkLine);
			for (size_t x=0; x<width; ++x) {
				int total = pTotalLine[x] - pWorkLine2[x] + pWorkLine[x];
				pDestLine[x] = (total * invLen) >> SHIFT;
				pTotalLine[x] = total;
			}
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine2, -workLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
		}
	}else {
		for (size_t x=0; x<width; ++x) {
			pTotalLine[x] = 0;
		}
		OffsetPtr(pSrcLine, -r * srcLineOffsetBytes);
		OffsetPtr(pWorkLine, -r * workLineOffsetBytes);
		pWorkLine2 = pWorkLine;
		for (int y=-r; y<=r; ++y) {
			horizontal.process(pSrcLine, pWorkLine);
			for (size_t x=0; x<width; ++x) {
				pTotalLine[x] += pWorkLine[x];
			}
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	for (size_t y=kys0; y<kye0; ++y) {
		horizontal.process(pSrcLine, pWorkLine);
		for (size_t x=0; x<width; ++x) {
			int total = pTotalLine[x] - pWorkLine2[x] + pWorkLine[x];
			pDestLine[x] = (total * invLen) >> SHIFT;
			pTotalLine[x] = total;
		}
		OffsetPtr(pSrcLine, srcLineOffsetBytes);
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		OffsetPtr(pWorkLine2, workLineOffsetBytes);
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	if (bBottom) {
		pWorkLine2 = pWork;
		OffsetPtr(pWorkLine2, (kye0 - r) * workLineOffsetBytes);
		pWorkLine = pWork;
		OffsetPtr(pWorkLine, (height - 1) * workLineOffsetBytes);
		for (size_t y=kye0; y<height; ++y) {
			for (size_t x=0; x<width; ++x) {
				int total = pTotalLine[x] - pWorkLine2[x] + pWorkLine[x];
				pDestLine[x] = (total * invLen) >> SHIFT;
				pTotalLine[x] = total;
			}
			OffsetPtr(pWorkLine2, workLineOffsetBytes);
			OffsetPtr(pWorkLine, -workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
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
	
	__forceinline void process(const uint8_t* __restrict pSrc, uint8_t* __restrict pWork) {
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
			__m128i left2 = _mm_mulhrs_epi16(left, mInvLen);
			__m128i right2 = _mm_mulhrs_epi16(right, mInvLen);
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
			
			__m128i mDest0 = _mm_mulhrs_epi16(mTotal0, mInvLen);
			__m128i mDest1 = _mm_mulhrs_epi16(mTotal1, mInvLen);
			
			__m128i mResult = _mm_packus_epi16(mDest0, mDest1);
			
			_mm_stream_si128(pDestLine+i, mResult);	// for single pass
//				pDestLine[i] = mResult;	// for multi pass
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
	
	int len = r * 2 + 1; // diameter
	int invLen = (1<<SHIFT) / len;
	if ((1<<SHIFT) % len) {
		++invLen;
	}

	int kys0 = 1;
	int kye0 = height;
	if (bTop) {
		kys0 = std::min<int>(height, 1 + r);
	}
	if (bBottom) {
		kye0 = std::max<int>(0, height - r);
	}
	
	const uint8_t* pSrcLine = pSrc;
	uint8_t* pWorkLine = pWork;
	uint8_t* pWorkLine2 = pWorkLine;
	uint8_t* pDestLine = pDest;
	
	HorizontalProcessor horizontal(width, r, invLen);
	VerticalProcessor vertical(width, invLen);
	
	if (bTop) {
		horizontal.process(pSrcLine, pWorkLine);
		for (size_t x=0; x<width; ++x) {
			pTotalLine[x] = pWorkLine[x];
		}
		OffsetPtr(pSrcLine, srcLineOffsetBytes);
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		for (size_t ky=1; ky<=r; ++ky) {
			horizontal.process(pSrcLine, pWorkLine);
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
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
		OffsetPtr(pWorkLine2, r * workLineOffsetBytes);
		
		for (size_t y=1; y<=r; ++y) {
			horizontal.process(pSrcLine, pWorkLine);
			vertical.process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pDestLine);
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine2, -workLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
		}
	}else {
		for (size_t x=0; x<width; ++x) {
			pTotalLine[x] = 0;
		}
		OffsetPtr(pSrcLine, -r * srcLineOffsetBytes);
		OffsetPtr(pWorkLine, -r * workLineOffsetBytes);
		pWorkLine2 = pWorkLine;
		for (int y=-r; y<=r; ++y) {
			horizontal.process(pSrcLine, pWorkLine);
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
			OffsetPtr(pSrcLine, srcLineOffsetBytes);
			OffsetPtr(pWorkLine, workLineOffsetBytes);
		}
		for (size_t x=0; x<width; ++x) {
			pDestLine[x] = (pTotalLine[x] * invLen) >> SHIFT;
		}
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	for (size_t y=kys0; y<kye0; ++y) {
		horizontal.process(pSrcLine, pWorkLine);
		vertical.process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pDestLine);
		OffsetPtr(pSrcLine, srcLineOffsetBytes);
		OffsetPtr(pWorkLine, workLineOffsetBytes);
		OffsetPtr(pWorkLine2, workLineOffsetBytes);
		OffsetPtr(pDestLine, destLineOffsetBytes);
	}
	
	if (bBottom) {
		pWorkLine2 = pWork;
		OffsetPtr(pWorkLine2, (kye0 - r) * workLineOffsetBytes);
		pWorkLine = pWork;
		OffsetPtr(pWorkLine, (height - 1) * workLineOffsetBytes);
		for (size_t y=kye0; y<height; ++y) {
			vertical.process((const __m128i*)pWorkLine2, (const __m128i*)pWorkLine, (__m128i*)pTotalLine, (__m128i*)pDestLine);
			OffsetPtr(pWorkLine2, workLineOffsetBytes);
			OffsetPtr(pWorkLine, -workLineOffsetBytes);
			OffsetPtr(pDestLine, destLineOffsetBytes);
		}
	}
	_mm_mfence();
}

} // namespace blur_1b
