#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "defines.h"

#include <stdlib.h>
#include <stdint.h>

typedef struct EmbeddingTable {
    uint8_t vocab_size;
    float *data;
} embedding_table_t;
// Una embedding table es una tabla donde cada token k está mapeado a un vector table[k]


embedding_table_t init_embeddings(uint8_t vocab_size);
float* get_embedding_from_id(embedding_table_t *table, uint8_t token);
#endif
