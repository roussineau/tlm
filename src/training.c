#include "training.h"
#include <math.h>
#include <float.h>
#include <stdio.h>

float cross_entropy_loss(const float *probs, uint8_t target_id){ 
    float p = fmaxf(probs[target_id], 1e-9f);
    return -logf(p);
}

// dlogits
void backward_logits(const float *probs, uint8_t target_id, float *dlogits, uint16_t vocab_size){
    for (int i = 0; i < vocab_size; i++){
        dlogits[i] = probs[i];
    }
    dlogits[target_id] -= 1;
}

// dW, db y dcontext
void backward_output_layer(const float *dlogits, const float *context, float *dcontext, float *dW, float *db, const float *W, uint16_t vocab_size){
    // Inicializo dW en 0
    for (int i = 0; i < vocab_size; i++){
        for (int j = 0; j < EMBEDDING_DIM; j++){
            dW[i * EMBEDDING_DIM + j] = 0;
        }
    }
    // Inicializo db en 0
    for (int i = 0; i < vocab_size; i++){
        db[i] = 0;
    }
    // Inicializo dcontext en 0
    for (int i = 0; i < EMBEDDING_DIM; i++){
        dcontext[i] = 0;
    }
    // Regla de la cadena
    for (int i = 0; i < vocab_size; i++){
        db[i] += dlogits[i];
        for (int j = 0; j < EMBEDDING_DIM; j++){
            dW[i * EMBEDDING_DIM + j] += dlogits[i] * context[j];
            dcontext[j] += dlogits[i] * W[i * EMBEDDING_DIM + j];
        }
    }
}

// dE
void backward_embeddings(const uint8_t *context_ids, const float *dcontext, embedding_table_t *emb, embedding_table_t *pos){
    float scale = 1.0f / CONTEXT_SIZE;

    for (int t = 0; t < CONTEXT_SIZE; t++){
        uint8_t token_id = context_ids[t];
        if (token_id == 0) continue;

        for (int j = 0; j < EMBEDDING_DIM; j++){
            float grad = scale * dcontext[j];
            emb->dE[token_id * EMBEDDING_DIM + j] += grad;
            pos->dE[t * EMBEDDING_DIM + j] += grad;
        }
    }
}

// Updates

void update_output_layer(layer_t *out, float learning_rate){
    int vocab_size = out->output_size;

    // W
    for (int i = 0; i < vocab_size * EMBEDDING_DIM; i++) {
        out->W[i] -= learning_rate * out->dW[i];
        out->dW[i] = 0.0f; // reset gradiente
    }

    // b
    for (int i = 0; i < vocab_size; i++) {
        out->b[i] -= learning_rate * out->db[i];
        out->db[i] = 0.0f;
    }
}

void update_embeddings(embedding_table_t *emb, embedding_table_t *pos, float learning_rate){
    int size = emb->size * EMBEDDING_DIM;

    for (int i = 0; i < size; i++) {
        emb->data[i] -= learning_rate * emb->dE[i];
        emb->dE[i] = 0.0f;
    }

    for (int i = 0; i < pos->size * EMBEDDING_DIM; i++) {
        pos->data[i] -= learning_rate * pos->dE[i];
        pos->dE[i] = 0.0f;
    }
}

// Entrenamiento

float train_step(embedding_table_t *emb, embedding_table_t *pos, layer_t *hidden, layer_t *out, uint8_t *context_ids, uint8_t target_id, float learning_rate){
    // Forward pass

    // Concatenar matriz EMBEDDINGS_DIM x CONTEXT_SIZE en vector 1 x (EMBEDDINGS_DIM x CONTEXT_SIZE)
    float context[EMBEDDING_DIM * CONTEXT_SIZE];
    embed_and_concatenate(emb, pos, context_ids, context);
    
    // Entrar a Hidden Layer
    // Entrada: vector 1 x (EMBEDDINGS_DIM x CONTEXT_SIZE)
    // W1: (EMBEDDINGS_DIM x CONTEXT_SIZE) x HIDDEN_DIM
    // b1: 1 x HIDDEN_DIM
    // Activacion: ReLU
    // Salida: vector 1 x HIDDEN_DIM
    float hidden_pre_activate[HIDDEN_DIM];
    linear_transform(hidden, context, hidden_pre_activate);

    float hidden_out[HIDDEN_DIM];
    relu(hidden_pre_activate, hidden_out, HIDDEN_DIM);
    
    // Entrar a Output Layer
    // Entrada: vector 1 x HIDDEN_DIM
    // WO: HIDDEN_DIM x vocab_size
    // bO: 1 x vocab_size
    // Activación: Softmax
    // Salida: vector de probabilidades 1 x vocab_size 

    float logits[out->output_size];
    linear_transform(out, hidden_out, logits);

    logits[0] = -FLT_MAX; // padding prohibido

    float probs[out->output_size];
    softmax(logits, probs, out->output_size);

    // Loss
    float loss = cross_entropy_loss(probs, target_id);

    // Backward pass
    float dlogits[out->output_size];
    backward_logits(probs, target_id, dlogits, out->output_size);

    float dcontext[EMBEDDING_DIM];
    backward_output_layer(dlogits, context, dcontext, out->dW, out->db, out->W, out->output_size);

    backward_embeddings(context_ids, dcontext, emb, pos);

    // Updates
    update_output_layer(out, learning_rate);
    update_embeddings(emb, pos, learning_rate);

    return loss;
}

void train(dataset_t *dataset, embedding_table_t *emb, embedding_table_t *pos, layer_t *hidden, layer_t *out){
    int epochs = 5;
    float lr = 0.05f;
    for (int e = 0; e < epochs; e++) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < dataset->num_samples; i++) {
            total_loss += train_step(emb, pos, hidden, out, dataset->inputs[i], dataset->targets[i], lr);
        }
        printf("Epoch %d | Loss promedio: %.4f\n", e, total_loss / dataset->num_samples);
    }
}
