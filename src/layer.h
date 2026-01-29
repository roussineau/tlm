#ifndef LAYER_H
#define LAYER_H

#include "defines.h"
#include "embeddings.h"
#include <stdint.h>

typedef struct Layer {
  uint16_t input_size;
  uint16_t output_size;
  float *W; // matriz de pesos (size x embed_dim)
  float *b; // vector de sesgo (size)
  float *dW;
  float *db;
} layer_t;

layer_t init_layer(uint16_t input_size, uint16_t output_size);

void linear_transform(layer_t *layer, float *context, float *out_logits);

void softmax(float *logit, float *probs, uint16_t n);

void relu(float *logit, float *out, uint16_t size);

uint8_t predict_next_token(embedding_table_t *emb, embedding_table_t *pos, layer_t *hidden, layer_t *out, uint8_t *context);

#endif
