#pragma once

#include <stdio.h>
#include <tchar.h>
#include <assert.h>
#include "stdint.h"

template <typename T>
__forceinline void OffsetPtr(T*& ptr, ptrdiff_t offsetBytes)
{
	ptr = (T*) ((const char*)ptr + offsetBytes);
}

#ifndef _MSC_VER

template <typename T, size_t N>
size_t __countof( T (&array)[N] )
{
    return N;
}

#endif

// http://stackoverflow.com/a/31981256/4699324

#ifdef _MSC_VER
#    if (_MSC_VER >= 1800)
#        define __alignas_is_defined 1
#    endif
#    if (_MSC_VER >= 1900)
#        define __alignof_is_defined 1
#    endif
#else
#    include <cstdalign>   // __alignas/of_is_defined directly from the implementation
#endif

#ifdef __alignas_is_defined
#    define ALIGN(X) alignas(X)
#else
#    pragma message("C++11 alignas unsupported :( Falling back to compiler attributes")
#    ifdef __GNUG__
#        define ALIGN(X) __attribute__ ((aligned(X)))
#    elif defined(_MSC_VER)
#        define ALIGN(X) __declspec(align(X))
#    else
#        error Unknown compiler, unknown alignment attribute!
#    endif
#endif

#ifdef __alignof_is_defined
#    define ALIGNOF(X) alignof(x)
#else
#    pragma message("C++11 alignof unsupported :( Falling back to compiler attributes")
#    ifdef __GNUG__
#        define ALIGNOF(X) __alignof__ (X)
#    elif defined(_MSC_VER)
#        define ALIGNOF(X) __alignof(X)
#    else
#        error Unknown compiler, unknown alignment attribute!
#    endif
#endif


#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

#ifndef _MSC_VER
#define __forceinline __attribute__((always_inline))
#endif



