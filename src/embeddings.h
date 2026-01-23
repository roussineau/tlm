#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "defines.h"

#include <stdlib.h>
#include <stdint.h>

typedef struct EmbeddingTable {
    uint16_t size;
    float *data;
    float *dE;
} embedding_table_t;
// Una embedding table es una tabla donde cada token k está mapeado a un vector table[k]

embedding_table_t init_id_embeddings(uint16_t vocab_size);

embedding_table_t init_pos_embeddings();

float* get_embedding_from(embedding_table_t *table, uint8_t num);

// Esta función genera los context_vector a partir de input que es un arreglo de IDs
void embed_and_aggregate(embedding_table_t* tokens, embedding_table_t* positions, uint8_t* input, float*output_context_vector);

#endif
