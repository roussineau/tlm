#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "defines.h"

#include <stdlib.h>
#include <stdint.h>

/* 1. TOKENIZACIÓN
    Tenemos que poder cargar texto y convertirlo en números:
        * Leer un archivo de texto
        * Encontrar los caracteres que aparecen
        * Darles un ID a cada uno
        * Convertir el texto en una secuencia de IDs
    Vamos a usar extended ASCII (0-255)
*/

typedef struct Vocab {
    uint16_t size; // El size se va a ir aumentando a medida que reconozcamos caracteres
    int16_t char_to_id[MAX_VOCAB]; // Todos los char van a tener ID = -1 al empezar, por eso necesitamos int16_t. Notar que si no se ocupa todo el vocabulario (256 caracteres), van a quedar -1s.
    uint8_t id_to_char[MAX_VOCAB];    
} vocab_t;

vocab_t vocab_init();

void add_new_char(vocab_t *v, uint8_t c);

void build_vocab_from_file(vocab_t *v, const char *filename);

void encode_file(vocab_t *v, const char *filename, uint8_t **ids_array, size_t *ids_array_length);

#endif