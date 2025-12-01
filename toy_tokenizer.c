#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

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

// Paso A: construir el vocabulario
#define MAX_VOCAB 256

typedef struct Vocab {
    uint8_t size; // El size se va a ir aumentando a medida que reconozcamos caracteres
    int16_t char_to_id[MAX_VOCAB]; // Todos los char van a tener ID -1 al empezar, por eso necesitamos int16_t. Notar que si no se ocupa todo el vocabulario (256 caracteres), van a quedar -1s.
    uint8_t id_to_char[MAX_VOCAB];    
} vocab_t;


vocab_t vocab_init(){
    vocab_t vocabulary;
    vocabulary.size = 0;
    memset(vocabulary.char_to_id, -1, sizeof(vocabulary.char_to_id)); // ID inicial = -1 para todos los caracteres
    return vocabulary;
}

void add_new_char(vocab_t *v, uint8_t c){
    if (v->char_to_id[c] != -1) return; // Ya lo teníamos en el vocabulario
    
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