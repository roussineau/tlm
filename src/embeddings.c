#include "embeddings.h"
#include "dataset.h"

#include <stdlib.h>

embedding_table_t init_id_embeddings(uint16_t vocab_size) {
    embedding_table_t table;
    table.size = vocab_size;

    table.data = malloc(vocab_size * EMBEDDING_DIM * sizeof(float));
    table.dE   = malloc(vocab_size * EMBEDDING_DIM * sizeof(float));

    // Inicializar embeddings aleatorios
    for (int i = 0; i < vocab_size * EMBEDDING_DIM; i++) {
        table.data[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        table.dE[i] = 0.0f;
    }

    // Embedding del padding (token 0) = vector nulo
    for (int j = 0; j < EMBEDDING_DIM; j++) {
        table.data[j] = 0.0f;
    }

    return table;
}

embedding_table_t init_pos_embeddings() {
    embedding_table_t table;
    table.size = CONTEXT_SIZE;

    table.data = malloc(CONTEXT_SIZE * EMBEDDING_DIM * sizeof(float));
    table.dE   = malloc(CONTEXT_SIZE * EMBEDDING_DIM * sizeof(float));

    // Inicializar embeddings posicionales aleatorios
    for (int i = 0; i < CONTEXT_SIZE * EMBEDDING_DIM; i++) {
        table.data[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        table.dE[i] = 0.0f;
    }

    return table;
}

float* get_embedding_from(embedding_table_t *table, uint8_t num){
    return &table->data[num * EMBEDDING_DIM];
}

void embed_and_aggregate(embedding_table_t *tokens, embedding_table_t *positions, uint8_t *input, float *out_context_vector){
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        out_context_vector[i] = 0;
    }

    float scale = 1.0f / CONTEXT_SIZE;

    for (int j = 0; j < CONTEXT_SIZE; j++) {
        if (input[j] == 0) continue; // El padding no aporta información
        float *token_embed = get_embedding_from(tokens, input[j]);
        float *pos_embed   = get_embedding_from(positions, j);

        for (int i = 0; i < EMBEDDING_DIM; i++) {
            out_context_vector[i] += scale * (token_embed[i] + pos_embed[i]);
        }
    }
}


