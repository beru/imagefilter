#pragma once

#include <string>

class Sym {
private:
	class SymImpl* pImpl_;
public:
	Sym();
	~Sym();
	
	std::string GetName(void* p);
};


