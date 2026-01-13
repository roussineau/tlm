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


    // Testing de embed_and_aggregate
    uint8_t inp[MAX_CONTEXT_SIZE] = {1};

    float out[EMBEDDING_DIM];

    embed_and_aggregate(&emb_table, &inp, &out);

    for(int i = 0; i < EMBEDDING_DIM; i++) {
        printf("%.3f\n", out[i]);
    }

    // 6. Cleanup mínimo
    free(ids);
    // (el resto lo liberamos más adelante)

    return 0;
}
