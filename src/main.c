#include <stdio.h>
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"
#include "dataset.h"
#include "embeddings.h"
#include "defines.h"

int main() {
    
    // 1. Construir vocabulario
    vocab_t vocab = vocab_init();
    build_vocab_from_file(&vocab, "data.txt");

    printf("Vocab size: %d\n", vocab.size);

    // 2. Tokenizar archivo
    uint8_t *ids = NULL;
    size_t ids_len = 0;
    encode_file(&vocab, "data.txt", &ids, &ids_len);

    // 3. Construir dataset
    dataset_t dataset = build_dataset_from(ids, ids_len);
    printf("Dataset con %zu muestras\n", dataset.num_samples);

    // 4. Inicializar embeddings
    embedding_table_t emb_table = init_embeddings(vocab.size);

    // 5. Testear embeddings de una fila del dataset
    size_t test_row = 0;
    printf("\nSample %zu\n", test_row);

    for (int i = 0; i < MAX_CONTEXT_SIZE; i++) {
        uint8_t token = dataset.inputs[test_row][i];
        float *vec = get_embedding_from_id(&emb_table, token);

        printf("Token ID %d → [", token);
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            printf("%+.3f ", vec[d]);
        }
        printf("]\n");
    }

    printf("Target ID: %d\n", dataset.targets[test_row]);

    // 6. Cleanup mínimo
    free(ids);
    // (el resto lo liberamos más adelante)

    return 0;
}
