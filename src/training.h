#ifndef TRAINING_H
#define TRAINING_H

#include "dataset.h"
#include "embeddings.h"
#include "layer.h"
#include <stdint.h>

float cross_entropy_loss(const float *probs, uint8_t target_id);

// dlogits
void backward_logits(const float *probs, uint8_t target_id, float *dlogits, uint16_t vocab_size);

// dWO, dbO y dhidden_out
void backward_output_layer(const float *dlogits, const float *hidden_out, float *dhidden_out, float *dW, float *db, const float *W, uint16_t vocab_size);

// dWH, dbH y dcontext
void backward_hidden_layer(const float *dhidden_out, const float *context, float *dcontext, float *dW, float *db, const float *W, uint16_t vocab_size);

// dE y dP
void backward_embeddings(const uint8_t *context_ids, const float *dcontext, embedding_table_t *emb, embedding_table_t *pos);

// Updates

void update_output_layer(layer_t *out, float learning_rate);

void update_embeddings(embedding_table_t *emb, embedding_table_t *pos, float learning_rate);

// Entrenamiento

float train_step(embedding_table_t *emb, embedding_table_t *pos, layer_t *hidden, layer_t *out, uint8_t *context_ids, uint8_t target_id, float learning_rate);

void train(dataset_t *dataset, embedding_table_t *emb, embedding_table_t *pos, layer_t *hidden, layer_t *out);

#endif
