# Roadmap del proyecto

Para hacer un modelo de lenguaje mínimo, necesitamos:

    1. Un mecanismo para pasar de texto a números (tokenización)
    2. Una representación numérica entrenable de esos tokens (embeddings)
    3. Un mecanismo para combinar información del contexto (atención)
    4. Un mecanismo para predecir el próximo token (softmax + logits)
    5. Algún método de balanceo (gradiente descendente)

De la primera parte se encarga el `tokenizer`, que convierte una secuencia de caracteres ASCII en una
secuencia de tokens.

De la segunda parte se va a encargar el `dataset`, que usa el tokenizer para leer un corpus, construye
pares (input, target), aplica context length y genera batches de ejemplos para entrenamiento.

Un modelo, finalmente, recibe un batch de secuencias ed IDs, aprende a mapearlas a un target