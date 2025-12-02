#include <stdio.h>
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"
#include "dataset.h"
#include "defines.h"

int main() {
    vocab_t vocab = vocab_init();

    uint8_t *tokenized_array = NULL;
    size_t tokenized_array_length = 0;

    // 1. Construir vocabulario
    build_vocab_from_file(&vocab, "data.txt");

    // 2. Tokenizar archivo
    encode_file(&vocab, "data.txt", &tokenized_array, &tokenized_array_length);

    printf("Leídos %zu IDs:\n", tokenized_array_length);
    for (size_t i = 0; i < tokenized_array_length; i++) {
        printf("%u ", tokenized_array[i]);
    }
    printf("\n\n");

    // 3. Construir dataset variable
    dataset_t ds = build_dataset_from(tokenized_array, tokenized_array_length);

    printf("Dataset generado con %zu muestras.\n", ds.num_samples);

    // 4. Imprimir las primeras 1500 muestras de debug
    size_t to_print = ds.num_samples < 1500 ? ds.num_samples : 1500;

    for (size_t i = 0; i < to_print; i++) {
        printf("Sample %zu: [", i);
        for (size_t j = 0; j < MAX_CONTEXT_SIZE; j++) {
            printf("%u", ds.inputs[i][j]);
            if (j + 1 < MAX_CONTEXT_SIZE) printf(", ");
        }
        printf("] -> %u\n", ds.targets[i]);
    }

    // 5. Liberar memoria
    for (size_t i = 0; i < ds.num_samples; i++) {
        free(ds.inputs[i]);
    }
    free(ds.inputs);
    free(ds.targets);
    free(tokenized_array);

    return 0;
}
