#include "output_layer.h"
#include "embeddings.h"

#include <stdint.h>

output_layer_t init_output_layer(uint8_t vocab_size){
    float *weights = malloc(sizeof(float) * EMBEDDING_DIM * vocab_size);
    float *bias = malloc(sizeof(float) * vocab_size);

    // Inicializar valores de la matriz de pesos
    for (int i = 0; i < EMBEDDING_DIM * vocab_size; i++){
        weights[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    }

    // Inicializar valores del vector de sesgo
    for (int i = 0; i < vocab_size; i++){
        bias[i] = 0.0f;
    }

    output_layer_t layer = {
        .vocab_size = vocab_size,
        .W = weights,
        .b = bias
    };

    return layer;
}

void compute_logits(output_layer_t *layer, float *context, float *output_logits) {
    // logit = Weights . context + bias

    for (int row = 0; row < layer->vocab_size; row++) {
        float sum = 0.0f;

        // Producto punto
        for (int col = 0; col < EMBEDDING_DIM; col++) {
            int w_index = (row * EMBEDDING_DIM) + col;
            sum += layer->W[w_index] * context[col];
        }

        // Sumar sesgo
        output_logits[row] = sum + layer->b[row];
    }
}

// uint8_t predict_next_token(embedding_table *emb, output_layer_t *out, uint8_t *context_ids){
    
// }
