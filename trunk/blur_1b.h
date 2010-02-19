#pragma once

namespace blur_1b {

struct Parameter {
	
	uint16_t width;
	uint16_t height;
	
	bool bTop;
	bool bBottom;
	
	const uint8_t* pSrc;
	ptrdiff_t srcLineOffsetBytes;
	
	uint8_t* pDest;
	ptrdiff_t destLineOffsetBytes;
	
	uint8_t* pWork;
	uint8_t* pWork2;
	ptrdiff_t workLineOffsetBytes;
	
	void* pTotal;
	void* pMinus;
	void* pPlus;
	
	uint8_t radius;
	
	uint8_t iterationCount;
	
};

// BoxFilter
void test_1(const Parameter& p);		// 1 pass naive implementation
void test_2(const Parameter& p);		// 1 pass naive implementation pointer optimized
void test_3(const Parameter& p);		// 2 pass optimization (horizontal -> vertical)
void test_4(const Parameter& p);		// 2 pass fixed point optimization
void test_5_h(const Parameter& p);		// horizontal slide in-out optimization
void test_5_v(const Parameter& p);		// vertical slide in-out optimization
void test_6_v(const Parameter& p);		// vertical slide in-out sequential memory access optimization
void test_7_h(const Parameter& p);		// horizontal slide in-out SSE optimization
void test_7_v(const Parameter& p);		// vertical slide in-out sequential memory access SSE optimization
void test_8(const Parameter& p);		// memory access optimized
void test_9(const Parameter& p);		// memory access further optimized
void test_10(const Parameter& p);		// SSE optimized

// TentFilter
void test_11(const Parameter& p);		// 

} // namespace blur_1b
