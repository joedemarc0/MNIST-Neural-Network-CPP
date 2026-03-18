#ifndef UTILS_H
#define UTILS_H

#include <string>


inline std::string ts(int value) { return std::to_string(value); }
inline std::string ts(size_t value) { return std::to_string(value); }
inline std::string ts(double value) { return std::to_string(value); }
inline std::string ts(bool value) { if (value) return std::string("true"); else return std::string("false"); }


#endif // UTILS_H