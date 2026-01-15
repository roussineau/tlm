#include <stdio.h>
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "tokenizer.h"
#include "dataset.h"
#include "embeddings.h"
#include "output_layer.h"
#include "defines.h"

int main() {
    srand(time(NULL));

    // 1. Construir vocabulario
    vocab_t vocab = vocab_init();
    build_vocab_from_file(&vocab, "data.txt");
    print_vocab(&vocab);

    // 2. Tokenizar archivo
    uint8_t *ids = NULL;
    size_t ids_len = 0;
    encode_file(&vocab, "data.txt", &ids, &ids_len);

    // 3. Construir dataset
    dataset_t dataset = build_dataset_from(ids, ids_len);
    printf("Dataset con %zu muestras\n", dataset.num_samples);

    // 4. Inicializar embeddings
    embedding_table_t emb_table = init_embeddings(vocab.size);

    // 5. Inicializar output layer (matriz de pesos y vector de sesgo)
    output_layer_t out = init_output_layer(vocab.size);
    
    // 6. Prueba de generación autorregresiva con un input elegido
    uint8_t context[MAX_CONTEXT_SIZE];
    memcpy(context, dataset.inputs[93000], MAX_CONTEXT_SIZE);

    printf("Input: [");
    for (int i = 0; i < MAX_CONTEXT_SIZE; i++){
        printf("%d", context[i]);
        if (!(i == MAX_CONTEXT_SIZE-1)){
            printf(", ");
        }
    }
    printf("]\n");

    int steps = 20;


    for (int i = 0; i < steps; i++) {
        uint8_t next = predict_next_token(&emb_table, &out, context);

        uint8_t ch = vocab.id_to_char[next];
        int16_t num = vocab.char_to_id[ch];
        printf("%d ", next);
        printf("%d ", num);
        printf("%c ", ch);

        printf("\n");
        // shift de la ventana
        for (int j = 0; j < MAX_CONTEXT_SIZE - 1; j++) {
            context[j] = context[j + 1];
        }
        context[MAX_CONTEXT_SIZE - 1] = next;
    }

    
    free(ids);

    return 0;
}


