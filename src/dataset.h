#ifndef DATASET_H
#define DATASET_H

#include "defines.h"

#include <stdlib.h>
#include <stdint.h>

typedef struct Dataset {
    size_t num_samples;
    uint8_t **inputs;
    uint8_t *targets;
} dataset_t;

dataset_t build_dataset(uint8_t *ids, size_t length);

#endif