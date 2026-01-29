#include "layer.h"
#include "embeddings.h"
#include "defines.h"

#include "training.h"

#include <float.h>
#include <math.h>

layer_t init_layer(uint16_t input_size, uint16_t output_size){
    float *weights = malloc(sizeof(float) * input_size * output_size);
    float *bias = malloc(sizeof(float) * output_size);
    float *dweights = malloc(sizeof(float) * input_size * output_size);
    float *dbias = malloc(sizeof(float) * output_size);

    // Inicializar valores de la matriz de pesos
    for (int i = 0; i < input_size * output_size; i++){
        weights[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f; // [-0,01; 0,01]
        dweights[i] = 0.0f;
    }

    // Inicializar valores del vector de sesgo
    for (int i = 0; i < output_size; i++){
        bias[i] = 0.0f;
        dbias[i] = 0.0f;
    }

    layer_t layer = {
        .input_size = input_size,
        .output_size = output_size,
        .W = weights,
        .b = bias,
        .dW = dweights,
        .db = dbias
    };

    return layer;
}

void softmax(float *logit, float *probs, uint16_t size){
    // Conseguir el máximo para corregir el overflow
    float max = logit[0];
    for (int i = 0; i < size; i++){
        if (logit[i] > max) {
            max = logit[i];
        } 
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++){
        // Exponenciar con desplazamiento 
        probs[i] = expf(logit[i] - max);
        // Sumar
        sum += probs[i];
    }

    // Normalizar
    for (int i = 0; i < size; i++){
        probs[i] /= sum;
    }
}

void relu(float *logit, float *out, uint16_t size){
    for (int i = 0; i < size; i++){
        out[i] = fmaxf(logit[i], 0.0f);
    }
}

void linear_transform(layer_t *layer, float *input, float *output) {
    for (int j = 0; j < layer->output_size; j++) {
        float sum = 0.0f;

        // Producto punto
        for (int i = 0; i < layer->input_size; i++) {
            int w_index = i * layer->output_size + j;
            sum += input[i] * layer->W[w_index];
        }

        // Sumar sesgo
        output[j] = sum + layer->b[j];
    }
}


uint8_t predict_next_token(embedding_table_t *tokens, embedding_table_t *positions, layer_t *hidden, layer_t *out, uint8_t *context_ids){
    float context[EMBEDDING_DIM * CONTEXT_SIZE];
    embed_and_concatenate(tokens, positions, context_ids, context);

    // Hidden
    float hidden_pre[HIDDEN_DIM];
    linear_transform(hidden, context, hidden_pre);

    float hidden_out[HIDDEN_DIM];
    relu(hidden_pre, hidden_out, HIDDEN_DIM);

    // Output
    float logits[out->output_size];
    linear_transform(out, hidden_out, logits);

    logits[0] = -FLT_MAX;

    float probs[out->output_size];
    softmax(logits, probs, out->output_size);

    // Argmax
    int id = 0;
    float max_prob = probs[0];
    for (int i = 1; i < out->output_size; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            id = i;
        }
    }

    return (uint8_t)id;
}
