#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include "defines.h"
#include "embeddings.h"
#include <stdint.h>

typedef struct OutputLayer {
    uint16_t vocab_size;
    float *W;  // matriz de pesos (vocab_size x embed_dim) 
    float *b;  // vector de sesgo (vocab_size)
    float *dW;
    float *db;
} output_layer_t;

output_layer_t init_output_layer(uint16_t vocab_size);

void compute_logits(output_layer_t *layer, float *context, float *out_logits);

void softmax(float *logit, float *probs, uint16_t n);

uint8_t predict_next_token(embedding_table_t *emb, output_layer_t *out, uint8_t *context);

#endif
