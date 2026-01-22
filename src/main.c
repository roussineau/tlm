#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "preprocess_file.h"
#include "tokenizer.h"
#include "dataset.h"
#include "embeddings.h"
#include "output_layer.h"
#include "training.h"
#include "defines.h"

int main(void) {
   srand((unsigned)time(NULL));

   // 1. Preprocesamiento
   preprocess_file("data.txt", "data_processed.txt");

   // 2. Vocabulario
   vocab_t vocab = vocab_init();
   build_vocab_from_file(&vocab, "data_processed.txt");
   printf("Vocab size: %d\n", vocab.size);

   // 3. Tokenización
   uint8_t *ids = NULL;
   size_t ids_len = 0;
   encode_file(&vocab, "data_processed.txt", &ids, &ids_len);

   // 4. Dataset
   dataset_t dataset = build_dataset_from(ids, ids_len);
   printf("Dataset con %zu muestras\n", dataset.num_samples);

   // 5. Modelo
   embedding_table_t emb = init_embeddings(vocab.size);
   output_layer_t out = init_output_layer(vocab.size);

   // 6. Entrenamiento
   train(&dataset, &emb, &out);

   // 7. Generación
   printf("\n=== Generación ===\n");
   uint8_t context[CONTEXT_SIZE];
   memcpy(context, dataset.inputs[0], CONTEXT_SIZE);

   int steps = 200;

   for (int i = 0; i < steps; i++) {
      uint8_t next = predict_next_token(&emb, &out, context);
      char ch = vocab.id_to_char[next];

      if (next >= vocab.size) {
         printf("[ERROR] token fuera de vocab: %d\n", next);
         break;
      }

      putchar(ch);

      // Shift del contexto
      memmove(context, context + 1, (CONTEXT_SIZE - 1) * sizeof(uint8_t));
      context[CONTEXT_SIZE - 1] = next;
   }

   putchar('.');
   putchar('\n');

   // 8. Cleanup
   free(ids);

   for (size_t i = 0; i < dataset.num_samples; i++) {
      free(dataset.inputs[i]);
   }
   free(dataset.inputs);
   free(dataset.targets);

   free(emb.data);
   free(emb.dE);

   free(out.b);
   free(out.db);
   free(out.W);
   free(out.dW);

   return 0;
}
