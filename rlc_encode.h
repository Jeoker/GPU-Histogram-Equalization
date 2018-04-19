#ifndef RLCENCODE
#define RLCENCODE
#include <vector>
#include <memory>
using namespace std;
typedef struct
{
    vector<unsigned char> *grey_value;
    vector<unsigned int> *number_count;
} rlcResult;

void encode(unsigned char *image, int size, rlcResult *rlc);

#endif
