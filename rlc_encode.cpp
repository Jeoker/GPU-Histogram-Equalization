#include <stdlib.h>
#include "rlc_encode.h"
#include <assert.h>
#include <stdio.h>
#include <iostream>

/* #define UNIT_TEST */

void encode(unsigned char *image, int size, rlcResult *rlc)
{
    vector<unsigned char> *grey_value = new vector<unsigned char>;
    vector<unsigned int> *number_count = new vector<unsigned int>;

    unsigned int count = 1;
    for (int i = 0; i < size; ++i)
    {
        if (i == 0 && i == size - 1)
        {
            (*grey_value).push_back(image[i]);
            (*number_count).push_back(count);
            continue;
        }

        if (i == size - 1)
        {
            if (image[i] == image[i - 1])
            {
                count += 1;
                (*grey_value).push_back(image[i - 1]);
                (*number_count).push_back(count);
            }
            else if (image[i] != image[i - 1])
            {
                (*grey_value).push_back(image[i - 1]);
                (*number_count).push_back(count);
                (*grey_value).push_back(image[i]);
                (*number_count).push_back(1);
            }
            continue;
        }

        if (i == 0)
        {
            continue;
        }
        else
        {
            if (image[i] == image[i - 1])
            {
                count += 1;
            }
            else
            {
                (*grey_value).push_back(image[i - 1]);
                (*number_count).push_back(count);
                count = 1;
            }
        }
    }

    for (int i = 1; i < (*number_count).size(); ++i)
    {
        (*number_count)[i] += (*number_count)[i - 1];
    }
    rlc->grey_value = grey_value;
    rlc->number_count = number_count;
}

#ifdef UNIT_TEST
/* Simple Unit test */
int main()
{
    unsigned char testInput[] = {'a', 'a', 'b', 'c', 'c',
                                 'd', 'd', 'a', 'e', 'a',
                                 'a', 'a', 'a', 'a', 'b'};
    int expected_num[] = {2, 3, 5, 7, 8, 9, 14, 15};
    unsigned char expected_grey[] = {'a', 'b', 'c', 'd', 'a', 'e', 'a', 'b'};

    rlcResult *rlc = (rlcResult *)malloc(sizeof(rlcResult));
    encode(testInput, 15, rlc);
    bool flag = false;

    for (int i = 0; i < (*(rlc->number_count)).size(); ++i)
    {
        if (expected_num[i] != (*(rlc->number_count))[i] || expected_grey[i] != (*(rlc->grey_value))[i])
        {
            cout << "error starts at " << i << endl;
            cout << "expected_num is " << expected_num[i] << endl;
            cout << "expected_grey is " << expected_grey[i] << endl;
            cout << "rlc_number_count is " << (*(rlc->number_count))[i] << endl;
            cout << "rlc_grey_value is " << (*(rlc->grey_value))[i] << endl;
            break;
        }
        if (i == 7)
        {
            flag = true;
        }
    }
    assert(flag);
    cout << "Unit test passed" << endl;
    free(rlc);
    return 0;
}
#endif
