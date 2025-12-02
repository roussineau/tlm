#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"
#include "dataset.h"
#include "defines.h"

int main(void) {
    vocab_t vocab = vocab_init();

    // 1. Construir vocabulario
    build_vocab_from_file(&vocab, "data.txt");
    printf("Vocab size = %u\n", vocab.size);

    // 2. Tokenizar archivo completo
    uint8_t *ids = NULL;
    size_t length = 0;

    encode_file(&vocab, "data.txt", &ids, &length);
    printf("Archivo tokenizado en %zu IDs.\n", length);

    // Imprimir algunos IDs
    printf("Primeros 20 IDs:\n");
    for (size_t i = 0; i < length && i < 20; i++) {
        printf("%u ", ids[i]);
    }
    printf("\n\n");

    // 3. Construir dataset
    dataset_t dataset = build_dataset(ids, length);

    printf("Dataset construido:\n");
    printf(" - num_samples = %zu\n", dataset.num_samples);
    printf(" - context size = %d\n", CONTEXT_SIZE);

    // 4. Imprimir las primeras 5 muestras para verificar
    size_t to_print = dataset.num_samples < 5 ? dataset.num_samples : 5;

    printf("\nPrimeras %zu muestras:\n", to_print);

    for (size_t s = 0; s < to_print; s++) {
        printf("Sample %zu: [", s);

        for (size_t j = 0; j < CONTEXT_SIZE; j++) {
            printf("%u", dataset.inputs[s][j]);
            if (j + 1 < CONTEXT_SIZE) printf(", ");
        }

        printf("] -> %u\n", dataset.targets[s]);
    }

    // 5. Liberar memoria
    for (size_t s = 0; s < dataset.num_samples; s++) {
        free(dataset.inputs[s]);
    }
    free(dataset.inputs);
    free(dataset.targets);
    free(ids);

    return 0;
}
