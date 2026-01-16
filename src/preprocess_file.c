#include <stdio.h>
#include <stdint.h>

int is_ascii_allowed(uint8_t c) {
    if (c == '\n') return 1;
    if (c >= 32 && c <= 126) return 1;
    return 0;
}

int map_utf8_diacritic(uint8_t first, uint8_t second, uint8_t *out) {
    if (first != 0xC3) return 0;

    switch (second) {
        case 0xA1: *out = 'a'; return 1; // á
        case 0xA9: *out = 'e'; return 1; // é
        case 0xAD: *out = 'i'; return 1; // í
        case 0xB3: *out = 'o'; return 1; // ó
        case 0xBA: *out = 'u'; return 1; // ú
        case 0xB1: *out = 'n'; return 1; // ñ

        case 0x81: *out = 'A'; return 1; // Á
        case 0x89: *out = 'E'; return 1; // É
        case 0x8D: *out = 'I'; return 1; // Í
        case 0x93: *out = 'O'; return 1; // Ó
        case 0x9A: *out = 'U'; return 1; // Ú
        case 0x91: *out = 'N'; return 1; // Ñ
    }
    return 0;
}

void preprocess_file(const char *input_path, const char *output_path) {
    FILE *in = fopen(input_path, "rb");
    FILE *out = fopen(output_path, "wb");

    if (!in || !out) {
        perror("Error abriendo archivo");
        return;
    }

    int b1;
    while ((b1 = fgetc(in)) != EOF) {
        uint8_t c1 = (uint8_t)b1;

        if (is_ascii_allowed(c1)) {
            fputc(c1, out);
            continue;
        }

        // Posible UTF-8 de dos bytes
        if (c1 == 0xC3) {
            int b2 = fgetc(in);
            if (b2 == EOF) break;

            uint8_t mapped;
            if (map_utf8_diacritic(c1, (uint8_t)b2, &mapped)) {
                fputc(mapped, out);
            }
            // si no mapea, se descarta
        }
        // otros bytes se ignoran
    }

    fclose(in);
    fclose(out);
}
