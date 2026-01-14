#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include "defines.h"

typedef struct OutputLayer {
    uint8_t vocab_size;
    float *W;  // matriz de pesos
    float *b;  // vector de sesgo
} output_layer_t;

output_layer_t init_output_layer(uint8_t vocab_size);

void compute_logits(output_layer_t *layer, float *context, float *out_logits);

#endif
