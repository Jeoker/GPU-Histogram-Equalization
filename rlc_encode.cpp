#include <omp.h>
#include <stdlib.h>
#include <rlc_encode.h>

rlcResult *encode(unsigned char *image, int width, int height) {
	vector<unsigned char> grey_value;
	vector<short> number_count;
	/* #pragma omp parallel for */ 
	/* Todo: openMp, need to rewrite delimters*/
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int index = i * width + j;
			int count = 1;
			/* deal with a special situation */
			if (j == 0 && j == width - 1) {
				grey_value.push_back(image[index]);
				number_count.push_back(count);
				continue;
			}
			/* if j is the last one of this row, we need stop */
			if (j == width - 1) {
				if (image[index] == image[index - 1]) {
					count += 1;	
					grey_value.push_back(image[index - 1]);
					number_count.push_back(count);
				} else if (image[index] != image[index = 1]) {
					grey_value.push_back(image[index - 1]);
					number_count.push_back(count);
					grey_value.push_back(image[index]);
					number_count.push_back(1);
				}
				continue;
			}

			if (j == 0) {
				continue;	
			} else {
				if (image[index] == image[index - 1]) {
					count += 1;	
				} else {
					grey_value.push_back(image[index - 1]);
					number_count.push_back(count);
					count = 1;
				}
			}
		} 
	}


	rlcResult *rel = (rlcResult*) malloc(sizeof(rlcResult));
	rel->grey_value = &grey_value;
	rel->number_grey = &number_count;
	rel->delimiters = &delimiters;
	return rel;
}

