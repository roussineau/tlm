#include "dataset.h"

dataset_t build_dataset_from(uint8_t *ids, size_t length){

    // Paso 1: reservar memoria
    size_t num_total_samples = length * MAX_CONTEXT_SIZE - ((MAX_CONTEXT_SIZE * (MAX_CONTEXT_SIZE + 1)) / 2);

    uint8_t **inputs = malloc(num_total_samples * sizeof(uint8_t*));
    for (size_t i = 0; i < num_total_samples; i++) {
        inputs[i] = malloc(MAX_CONTEXT_SIZE * sizeof(uint8_t));
    }

    uint8_t *targets = malloc(num_total_samples * sizeof(uint8_t));


    // Paso 2: empezar a llenar los inputs y targets
    size_t row = 0;

    size_t current_context_size = 1;
    while (current_context_size <= MAX_CONTEXT_SIZE) {

        size_t num_samples = length - current_context_size;

        for(size_t sample = 0; sample < num_samples; sample++) {

            size_t pad = MAX_CONTEXT_SIZE - current_context_size;
            // Padding
            for (size_t i = 0; i < pad; i++) {
                inputs[row][i] = 0;
            }
            // IDs reales
            for (size_t i = 0; i < current_context_size; i++) {
                inputs[row][pad + i] = ids[sample + i];
            }

            // Target
            targets[row] = ids[sample + current_context_size];

            row++;
        }

        current_context_size++;
    }

    dataset_t dataset = {
        .num_samples = num_total_samples,
        .inputs = inputs,
        .targets = targets
    };

    return dataset;
}