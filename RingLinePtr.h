#pragma once

template <typename PtrT>
class RingLinePtr {
private:
	const size_t size;
	int idx;
	PtrT pCur;
	const ptrdiff_t lineOffsetBytes;
	PtrT const pFirst;
	PtrT const pLast;
public:
	RingLinePtr(
		size_t size,
		size_t curIdx,
		PtrT pCur,
		ptrdiff_t lineOffsetBytes
	)
		:
		size(size),
		idx(curIdx),
		pCur(pCur),
		lineOffsetBytes(lineOffsetBytes),
		pFirst( (PtrT const) (((uint8_t*)pCur) + curIdx * -lineOffsetBytes) ),
		pLast( (PtrT const) (((uint8_t*)pCur) + (size - 1 - curIdx) * lineOffsetBytes) )
	{
		assert(curIdx < size);
	}
	
	RingLinePtr& operator = (const RingLinePtr& p) {
		assert(size == p.size);
		assert(pFirst == p.pFirst);
		assert(lineOffsetBytes == p.lineOffsetBytes);
		assert(pLast == p.pLast);
		idx = p.idx;
		pCur = p.pCur;
		return *this;
	}
	
	RingLinePtr& moveNext() {
		assert(idx < size);
		++idx;
		OffsetPtr(pCur, lineOffsetBytes);
		if (idx == size) {
			idx = 0;
			pCur = pFirst;
		}
		return *this;
	}
	
	RingLinePtr& movePrev() {
		assert(idx < size);
		if (idx == 0) {
			assert(pCur == pFirst);
			idx = size - 1;
			pCur = pLast;
		}else {
			--idx;
			OffsetPtr(pCur, -lineOffsetBytes);
		}
		return *this;
	}

	RingLinePtr& move(ptrdiff_t offset) {
		idx += offset;
		while (idx < 0) {
			idx += size;
		}
		while (idx > size) {
			idx -= size;
		}
		pCur = pFirst;
		OffsetPtr(pCur, idx * lineOffsetBytes);
		return *this;
	}
	
	template <typename PT>
	operator PT () const {
		return (PT) pCur;
	}

	operator PtrT () const {
		return pCur;
	}

};

