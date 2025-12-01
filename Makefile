# Makefile para toy_tokenizer
CC := gcc
CFLAGS := -std=c11 -O2 -Wall -Wextra -Wpedantic
TARGET := toy_tokenizer
SRCS := toy_tokenizer.c

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

# Ejecutar el programa — espera que exista data.txt en el directorio actual
run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
