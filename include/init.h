/**
 * Simple Init Class Implementation
 * 
 * Weight Initialization Functions are implemented in matrix.cpp however this allows variabilization (if that is a word)
 * of the weight initialization - weight initialization can be chosen through the lossType variable now.
 */

#ifndef INIT_H
#define INIT_H


enum class InitType { NONE, RANDOM, XAVIER, HE };

#endif // INIT_H