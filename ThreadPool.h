#pragma once

#include <windows.h>
#include <process.h>

template <typename T>
class Threads
{
public:
	Threads();
	~Threads();
	
	bool SetUp(unsigned char nThreads);
	typedef void (*JobFuncPtr)(const T& param);
	void Start(JobFuncPtr pFunc, const T* pParams);
	bool Join(int timeOutMilliseconds = -1);
	bool Shutdown();
	
private:
	
	JobFuncPtr pFunc_;
	const T* pParams_;
	
	bool setUpped_;
	static const unsigned char MAX_THREAD_COUNT = 255;
	
	struct ThreadProcCallerInfo
	{
		Threads* pThis;
		unsigned char slotId;
	};
	ThreadProcCallerInfo threadProcCallerInfos_[ MAX_THREAD_COUNT ];
	static unsigned int __stdcall ThreadProcCaller(void* param) {
		const ThreadProcCallerInfo& info = * (const ThreadProcCallerInfo*)(param);
		info.pThis->ThreadProc(info.slotId);
		return 0;
	}
	void ThreadProc(unsigned char slotId);
	
	unsigned char nThreads_;
	HANDLE hThreads_[ MAX_THREAD_COUNT ];
	unsigned int threadIds_[ MAX_THREAD_COUNT ];
	
	HANDLE hShutdownEvent_;
	
	HANDLE hBeginEvents_[ MAX_THREAD_COUNT ];
	HANDLE hEndedEvents_[ MAX_THREAD_COUNT ];

};

template <typename T>
Threads<T>::Threads()
	:
	setUpped_(false)
{
}

template <typename T>
Threads<T>::~Threads()
{
	Join();
	Shutdown();
}

template <typename T>
bool Threads<T>::SetUp(unsigned char nThreads)
{
	if (setUpped_) {
		return false;
	}
	
	hShutdownEvent_ = CreateEvent(0, TRUE, FALSE, 0);
	
	nThreads_ = nThreads;
	for (size_t i=0; i<nThreads_; ++i) {
		ThreadProcCallerInfo& info = threadProcCallerInfos_[i];
		info.pThis = this;
		info.slotId = (unsigned char)i;
		HANDLE hThread = (HANDLE) _beginthreadex(NULL, 0, Threads::ThreadProcCaller, &info, 0, &threadIds_[i]);
		hThreads_[i] = hThread;
		
		hBeginEvents_[i] = CreateEvent(0, TRUE, FALSE, 0);
		hEndedEvents_[i] = CreateEvent(0, TRUE, FALSE, 0);
	}
	
	setUpped_ = true;
	return true;
}

template <typename T>
void Threads<T>::ThreadProc(unsigned char slotId)
{
	HANDLE hEndedEvent = hEndedEvents_[slotId];
	HANDLE hBeginEvent = hBeginEvents_[slotId];
	HANDLE hWaitEvents[2];
	hWaitEvents[0] = hShutdownEvent_;
	hWaitEvents[1] = hBeginEvent;
	while (1) {
		DWORD ret = WaitForMultipleObjects(2, hWaitEvents, FALSE, INFINITE);
		switch (ret - WAIT_OBJECT_0) {
		case 0:
			return;
		case 1:
			assert(pFunc_);
			pFunc_(pParams_[slotId]);
			ResetEvent(hBeginEvent);
			SetEvent(hEndedEvent);
		}
	}
}

template <typename T>
void Threads<T>::Start(JobFuncPtr pFunc, const T* pParams)
{
	// TODO: check if threads are already started
	
	pFunc_ = pFunc;
	pParams_ = pParams;
	for (size_t i=0; i<nThreads_; ++i) {
		ResetEvent(hEndedEvents_[i]);
		SetEvent(hBeginEvents_[i]);
	}
}

template <typename T>
bool Threads<T>::Join(int timeOutMilliseconds = -1)
{
	DWORD result = WaitForMultipleObjects(nThreads_, hEndedEvents_, TRUE, timeOutMilliseconds);
	return WAIT_TIMEOUT != result;
}

template <typename T>
bool Threads<T>::Shutdown()
{
	if (!setUpped_) {
		return false;
	}

	SetEvent(hShutdownEvent_);
	WaitForMultipleObjects(nThreads_, hThreads_, TRUE, INFINITE);
	for (size_t i=0; i<nThreads_; ++i) {
		CloseHandle(hThreads_[i]);
		CloseHandle(hBeginEvents_[i]);
		CloseHandle(hEndedEvents_[i]);
	}
	CloseHandle(hShutdownEvent_);
	
	setUpped_ = false;
	return true;
}

