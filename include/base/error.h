#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <assert.h>

#define Error(message) \
  do { \
    std::cerr << "Error in " << __FILE__ << " line " << __LINE__ << ": " << message << std::endl; \
    assert(0); \
  } while (false)

#define Assert(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__ << " line " << __LINE__ << ": " << message << std::endl; \
      assert(0); \
      } \
    } while (false)

#endif