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
	ptrdiff_t workLineOffsetBytes;
	
	int16_t* pTotalLine;
	
	uint8_t radius;
	
};

void test_1(const Parameter& p);
void test_2(const Parameter& p);
void test_3(const Parameter& p);
void test_4(const Parameter& p);
void test_5_h(const Parameter& p);
void test_5_v(const Parameter& p);
void test_6_v(const Parameter& p);
void test_7_h(const Parameter& p);
void test_7_v(const Parameter& p);
void test_8(const Parameter& p);
void test_9(const Parameter& p);
void test_10(const Parameter& p);

} // namespace blur_1b
