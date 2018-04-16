#ifndef RLCENCODE
#define RLCENCODE
#include <vector>
using namespace std;
typedef struct {
	vector<unsigned char> *grey_value;
	vector<short> *number_grey;
} rlcResult;

rlcResult *encode(unsigned char *image, int width, int height);

#endif
