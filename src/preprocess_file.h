#ifndef PREPROCESS_FILE_H
#define PREPROCESS_FILE_H

#include <stdint.h>

int is_ascii_allowed(uint8_t c);

int map_utf8_diacritic(uint8_t first, uint8_t second, uint8_t *out);

void preprocess_file(const char *input_path, const char *output_path);

#endif
