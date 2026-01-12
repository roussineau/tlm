#include "embeddings.h"
#include "dataset.h"

#include <stdlib.h>

embedding_table_t init_embeddings(uint8_t vocab_size) {
    embedding_table_t table;
    table.vocab_size = vocab_size;
    table.data = malloc(vocab_size * EMBEDDING_DIM * sizeof(float));

    // Embedding del padding (token 0) = vector nulo
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        table.data[i] = 0.0f;
    }

    // Los embeddings van a estar todos uno detrás de otro, de forma tal que
    // el embedding del token i va de data[i*EMBEDDING_DIM] hasta data[i*EMBEDDING_DIM + EMBEDDING_DIM-1]
    for(int i = EMBEDDING_DIM; i < vocab_size * EMBEDDING_DIM; i++) {
        table.data[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    }

    return table;
}

float* get_embedding_from_id(embedding_table_t *table, uint8_t token_id){
    return &table->data[token_id * EMBEDDING_DIM];
}

float** lookup_embeddings(embedding_table_t *table, uint8_t *input){
    // lookup es un arreglo MAX_CONTEXT_SIZE punteros a float
    // De esta forma transformamos un input de un vector de IDs enteros a un vector de embeddings
    float** lookup = malloc(sizeof(float*) * MAX_CONTEXT_SIZE);

    for(int i = 0; i < MAX_CONTEXT_SIZE; i++) {
        lookup[i] = get_embedding_from_id(table, input[i]);
    }

    return lookup;
}

