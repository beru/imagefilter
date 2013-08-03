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
	void* pModi;
	
	uint8_t radius;
	
	uint8_t iterationCount;
	
};

// BoxFilter
void memory_copy1(const Parameter& p);
void memory_copy2(const Parameter& p);
void memory_copy3(const Parameter& p);

void test_1(const Parameter& p);		// 1 pass naive implementation
void test_2(const Parameter& p);		// 1 pass naive implementation pointer optimization
void test_3(const Parameter& p);		// 2 pass optimization (horizontal -> vertical)
void test_4(const Parameter& p);		// 2 pass fixed point optimization
void test_5_h(const Parameter& p);		// horizontal slide in-out optimization
void test_5_v(const Parameter& p);		// vertical slide in-out optimization
void test_6_v(const Parameter& p);		// vertical slide in-out sequential memory access optimization
void test_7_h(const Parameter& p);		// test_5 SSE optimization
void test_7_v(const Parameter& p);		// test_6 SSE optimization
void test_8(const Parameter& p);		// memory access optimization
void test_9(const Parameter& p);		// memory access further optimization
void test_10(const Parameter& p);		// test_9 SSE optimization
void test_11(const Parameter& p);		// fused horizontal & vertical
void test_12(const Parameter& p);		// test_11 SSE3 optimization

// TentFilter
void test_20(const Parameter& p);		// C implementation
void test_21(const Parameter& p);		// SSE optimization

} // namespace blur_1b
