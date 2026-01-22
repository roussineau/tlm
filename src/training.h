#ifndef TRAINING_H
#define TRAINING_H

#include "embeddings.h"
#include "output_layer.h"
#include "dataset.h"
#include <stdint.h>

float cross_entropy_loss(const float *probs, uint8_t target_id);

// dlogits
void backward_logits(
    const float *probs,
    uint8_t target_id,
    float *dlogits,
    uint16_t vocab_size
);

// dW, db y dcontext
void backward_output_layer(
    const float *dlogits,
    const float *context,
    float *dcontext,
    float *dW,
    float *db,
    const float *W,
    uint16_t vocab_size
);

// dE
void backward_embeddings(
    const uint8_t *context_ids,
    const float *dcontext,
    embedding_table_t *emb
);

// Updates

void update_output_layer(
    output_layer_t *out,
    float learning_rate
);

void update_embeddings(
    embedding_table_t *emb,
    float learning_rate
);

// Entrenamiento

float train_step(
    embedding_table_t *emb,
    output_layer_t *out,
    uint8_t *context_ids,
    uint8_t target_id,
    float learning_rate
);

void train(
    dataset_t *dataset,
    embedding_table_t *emb,
    output_layer_t *out
);

#endif

