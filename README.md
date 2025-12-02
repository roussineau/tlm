# Roadmap del proyecto

Para hacer un modelo de lenguaje mínimo, necesitamos:

    1. Un mecanismo para pasar de texto a números (tokenización)
    2. Una representación numérica entrenable de esos tokens (embeddings)
    3. Un mecanismo para combinar información del contexto (atención)
    4. Un mecanismo para predecir el próximo token (softmax + logits)
    5. Algún método de balanceo (gradiente descendente)

De la primera parte se encarga el `tokenizer`, que convierte una secuencia de caracteres ASCII en una
secuencia de tokens.

El código de `dataset` se encarga de representar ese archivo tokenizado como secuencias entrenables.
La traducción tiene la forma `inputs[i]` = `targets[i]`, donde cada lista del arreglo `inputs` tiene
longitud `CONTEXT_LENGTH`.