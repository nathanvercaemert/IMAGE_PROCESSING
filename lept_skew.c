#include <stdio.h>
#include <stdlib.h>
#include "allheaders.h"

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: lept_skew image [sweeprange]\n");
        return 2;
    }

    float sweeprange = (argc == 3) ? (float)atof(argv[2]) : 10.0f;

    PIX *pixs = pixRead(argv[1]);
    if (!pixs) {
        fprintf(stderr, "could not read image\n");
        return 1;
    }

    l_float32 angle = 0.0f;
    l_float32 conf = 0.0f;

    PIX *pixd = pixDeskewGeneral(
        pixs,
        0,              /* default sweep reduction */
        sweeprange,     /* search range in each direction */
        0.0,            /* default sweep delta */
        0,              /* default binary-search reduction */
        0,              /* default threshold */
        &angle,
        &conf
    );

    if (!pixd) {
        fprintf(stderr, "deskew failed\n");
        pixDestroy(&pixs);
        return 1;
    }

    printf("{\"angle\": %.4f, \"confidence\": %.4f}\n", angle, conf);

    pixDestroy(&pixd);
    pixDestroy(&pixs);
    return 0;
}
