#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* ROADMAP
    Para hacer un modelo de lenguaje mínimo, necesitamos:
        1. Un mecanismo para pasar de texto a números (tokenización)
        2. Una representación numérica entrenable de esos tokens (embeddings)
        3. Un mecanismo para combinar información del contexto (atención)
        4. Un mecanismo para predecir el próximo token (softmax + logits)
        5. Algún método de balanceo (gradiente descendente)
*/

/* 1. TOKENIZACIÓN
    Tenemos que poder cargar texto y convertirlo en números:
        * Leer un archivo de texto
        * Encontrar los caracteres que aparecen
        * Darles un ID a cada uno
        * Convertir el texto en una secuencia de IDs
    Vamos a usar extended ASCII (0-255)
*/


#define MAX_VOCAB 256

typedef struct Vocab {
    uint8_t size; // El size se va a ir aumentando a medida que reconozcamos caracteres
    uint16_t char_to_id[256]; // Todos los char van a tener ID -1 al empezar, por eso necesitamos uint16_t
    uint8_t id_to_char[MAX_VOCAB];    
} vocab_t;


vocab_t vocab_init(){
    vocab_t vocabulary;
    vocabulary.size = 0;
    memset(vocabulary.char_to_id, -1, sizeof(vocabulary.char_to_id)); // ID inicial = -1 para todos los caracteres
    return vocabulary;
}

void add_new_char(vocab_t *vocabulary, uint8_t character){
    if (vocabulary->char_to_id[character] != -1){
        return; // Ya lo teníamos en el vocabulario
    }
    uint8_t new_id = vocabulary->size;
    vocabulary->char_to_id[character] = new_id;
    vocabulary->id_to_char[new_id] = character;

    vocabulary->size++;
}






int main() {


    return 0;
}