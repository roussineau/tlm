#include "output_layer.h"
#include "embeddings.h"
#include "defines.h"

#include "training.h"

#include <float.h>
#include <math.h>

output_layer_t init_output_layer(uint16_t vocab_size){
    float *weights = malloc(sizeof(float) * EMBEDDING_DIM * vocab_size);
    float *bias = malloc(sizeof(float) * vocab_size);
    float *dweights = malloc(sizeof(float) * EMBEDDING_DIM * vocab_size);
    float *dbias = malloc(sizeof(float) * vocab_size);

    // Inicializar valores de la matriz de pesos
    for (int i = 0; i < EMBEDDING_DIM * vocab_size; i++){
        weights[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f; // [-0,01; 0,01]
        dweights[i] = 0.0f;
    }

    // Inicializar valores del vector de sesgo
    for (int i = 0; i < vocab_size; i++){
        bias[i] = 0.0f;
        dbias[i] = 0.0f;
    }

    output_layer_t layer = {
        .vocab_size = vocab_size,
        .W = weights,
        .b = bias,
        .dW = dweights,
        .db = dbias
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

void softmax(float *logit, float *probs, uint16_t vocab_size){
    // Conseguir el máximo para corregir el overflow
    float max = logit[0];
    for (int i = 0; i < vocab_size; i++){
        if (logit[i] > max) {
            max = logit[i];
        } 
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++){
        // Exponenciar con desplazamiento 
        probs[i] = expf(logit[i] - max);
        // Sumar
        sum += probs[i];
    }

    // Normalizar
    for (int i = 0; i < vocab_size; i++){
        probs[i] /= sum;
    }
}

uint8_t predict_next_token(embedding_table_t *emb, output_layer_t *out, uint8_t *context_ids){
    // La ventana de IDs que usamos es context_ids
    float context_vector[EMBEDDING_DIM];
    embed_and_aggregate(emb, context_ids, context_vector);
    
    float logits[out->vocab_size];
    compute_logits(out, context_vector, logits);

    logits[0] = -FLT_MAX; // Enmascaramos el token 0 porque es reservado

    float probs[out->vocab_size];
    softmax(logits, probs, out->vocab_size);

    float max_prob = probs[0];
    int id = 0;
    for (int i = 1; i < out->vocab_size; i++){
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            id = i;
        }
    }

    return (uint8_t)id;
}
