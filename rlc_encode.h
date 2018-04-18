#ifndef RLCENCODE
#define RLCENCODE
#include <vector>
using namespace std;
typedef struct {
	vector<unsigned char> grey_value;
	vector<unsigned int> number_count;
} rlcResult;

rlcResult *encode(unsigned char *image, int size, rlcResult* rlc);

#endif
