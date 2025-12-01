#include "tokenizer.h"

#include <stdio.h>
#include <string.h>

// Paso A: construir el vocabulario
vocab_t vocab_init(){
    vocab_t vocabulary;
    vocabulary.size = 0;
    memset(vocabulary.char_to_id, -1, sizeof(vocabulary.char_to_id)); // ID inicial = -1 para todos los caracteres
    return vocabulary;
}

void add_new_char(vocab_t *v, uint8_t c){
    if (v->char_to_id[c] != -1) return; // Ya lo teníamos en el vocabulario
    
    if (v->size >= MAX_VOCAB) return; // No pasarnos del buffer

    uint8_t new_id = v->size;
    v->char_to_id[c] = new_id;
    v->id_to_char[new_id] = c;
    v->size++;
}

// Paso B: leer el archivo y poblar el vocabulario
// Como estamos trabajando con extended ASCII, nuestro vocabulario es de a bytes (0-255).
// Vamos a abrir el archivo en modo de lectura binaria, leerlo byte a byte, y en cada lectura llamar a `add_new_char`

void build_vocab_from_file(vocab_t *v, const char *filename){
    FILE *stream = fopen(filename, "rb"); // Stream de caracteres extended ASCII, modo de lectura binaria
    if (!stream) return;

    int c; // Variable que va a guardar el byte leído, o valer -1 si se trata de un End Of File
    while ((c = fgetc(stream)) != EOF) {
        add_new_char(v, (uint8_t)c);
    }

    fclose(stream);
}

// Paso C: Tokenizar el texto en IDs
void encode_file(vocab_t *v, const char *filename, uint8_t **ids_array, size_t *ids_array_length){
    // En la dirección de ids_array vamos a guardar un puntero al arreglo de IDs
    // O sea que vamos a querer modificar el valor al que apunta para agregar memoria dinámica
    FILE *stream1 = fopen(filename, "rb");
    if (!stream1) return;

    size_t array_size = 0;
    int c1;
    while ((c1 = fgetc(stream1)) != EOF) {
        array_size++;
    }

    fclose(stream1); // Primera lectura completada

    FILE *stream2 = fopen(filename, "rb");
    if (!stream2) return;
    
    *ids_array = malloc(array_size * sizeof(uint8_t));

    int c2;
    size_t i = 0;
    while ((c2 = fgetc(stream2)) != EOF) {
        (*ids_array)[i] = v->char_to_id[c2];
        i++;
    }

    *ids_array_length = array_size;

    fclose(stream2); // Segunda lectura completada
}
