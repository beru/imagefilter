#include "sym.h"

#include <windows.h>
#include <Dbghelp.h>
#pragma comment(lib, "Dbghelp.lib")

/*
#pragma comment(lib, "msvcrtd.lib")
// msvcmrtd.lib
extern "C" {
extern char* __unDName(char *,const char*,int,int,int,unsigned short int);
extern char* __unDNameEx(char *,const char*,int,int,int,void *,unsigned short int);
}
*/

class SymImpl {
private:
	HANDLE hProcess_;
public:
	SymImpl() {
		// copied from xtrace
		
		DWORD symOptions = SymGetOptions();
		symOptions |= SYMOPT_DEFERRED_LOADS; 
		symOptions &= ~SYMOPT_UNDNAME;

		SymSetOptions(symOptions);

		hProcess_ = GetCurrentProcess();
		BOOL bInited = SymInitialize(
			hProcess_,
			NULL,
			TRUE
		);
		int hoge = 0;
	}
	
	~SymImpl() {
		SymCleanup(hProcess_);
	}
	
	std::string GetName(void* p) {
        DWORD64 displacement64;
        ULONG64 buffer[(sizeof(SYMBOL_INFO) +
            MAX_SYM_NAME * sizeof(TCHAR) +
            sizeof(ULONG64) - 1) /
            sizeof(ULONG64)];
        PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)buffer;

        pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        pSymbol->MaxNameLen = MAX_SYM_NAME;
		if (SymFromAddr(hProcess_, (DWORD64)p, 0, pSymbol)) {
			//char* pTempName = __unDNameEx(0, pSymbol->Name, 0, (int)malloc, (int)free, 0, UNDNAME_COMPLETE);
			//if (pTempName) {
			//	std::string ret = pTempName;
			//	free(pTempName);
			//	return ret;
			//}else {
			//	return "";
			//}
			const char* pQ = strchr(pSymbol->Name, '?');
			if (!pQ) {
				return pSymbol->Name;
			}
			char buff[MAX_SYM_NAME];
			strcpy(buff, pQ);
			int len = strlen(buff);
			if (buff[len-1] == ')') {
				buff[len-1] = 0;
			}
			char demangled[MAX_SYM_NAME];
			if (
				UnDecorateSymbolName(
					buff,
					demangled,
					MAX_SYM_NAME-1, 
					UNDNAME_NAME_ONLY
				)
			) {
				return demangled;
			}
		}
		return "Sym::GetName failed";
	}

};

Sym::Sym() {
	pImpl_ = new SymImpl();
}

Sym::~Sym() {
	delete pImpl_;
}

std::string Sym::GetName(void* p) {
	return pImpl_->GetName(p);
}

