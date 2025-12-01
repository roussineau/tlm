#include <stdio.h>
#include "tokenizer.h"

int main() {
    vocab_t vocab = vocab_init();

    uint8_t *tokenized_array = NULL;
    size_t tokenized_array_length = 0;

    build_vocab_from_file(&vocab, "data.txt");

    encode_file(&vocab, "data.txt", &tokenized_array, &tokenized_array_length);

    printf("Leídos %zu IDs:\n", tokenized_array_length);
    for (size_t i = 0; i < tokenized_array_length; i++) {
        printf("%u ", tokenized_array[i]);
    }

    printf("\n");

    free(tokenized_array);
    return 0;
}