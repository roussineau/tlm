#include "dataset.h"

dataset_t build_dataset(uint8_t *ids, size_t length){
    size_t num_samples = length - CONTEXT_SIZE;

    // RESERVAR MEMORIA:
    // Inputs:
    // Reservo array de punteros a filas
    uint8_t **inputs = malloc(num_samples * sizeof(uint8_t*));

    // Cada fila tiene CONTEXT_SIZE bytes
    for (size_t i = 0; i < num_samples; i++) {
        inputs[i] = malloc(CONTEXT_SIZE * sizeof(uint8_t));
    }

    // Targets:
    uint8_t *targets = malloc(num_samples * sizeof(uint8_t));

    // LLENADO:
    for (size_t sample = 0; sample < num_samples; sample++) {
        for (size_t i = 0; i < CONTEXT_SIZE; i++) {
            inputs[sample][i] = ids[sample + i];
        }
        targets[sample] = ids[sample + CONTEXT_SIZE];
    }

    dataset_t dataset = {
        .num_samples = num_samples,
        .inputs = inputs,
        .targets = targets
    };

    return dataset;
}

