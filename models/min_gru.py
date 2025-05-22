import tensorflow as tf
import numpy as np

# ========== Minimized GRU Model ==========
class MinGRUCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.z_gate = tf.keras.layers.Dense(hidden_size, activation='sigmoid')
        self.h_tilde = tf.keras.layers.Dense(hidden_size)
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def precompute(self, x):
        """
        Precompute a = (1 - z_t) and b = z_t * h̃_t for all time steps in parallel.
        x: [B, T, D]
        Returns:
            a: [B, T, H]
            b: [B, T, H]
        """
        z = self.z_gate(x)         # [B, T, H]
        h_hat = self.h_tilde(x)    # [B, T, H]
        a = 1.0 - z
        b = z * h_hat
        return a, b

def parallel_scan(a, b, h0):
    """
    Performs h_t = a_t * h_{t-1} + b_t using a scan (sequential version for simplicity).
    a: [B, T, H]
    b: [B, T, H]
    h0: [B, H]
    Returns:
        h: [B, T, H]
    """
    B, T, H = tf.shape(a)[0], tf.shape(a)[1], tf.shape(a)[2]
    h_list = tf.TensorArray(dtype=tf.float32, size=T)
    h_t = h0  # initial hidden state

    for t in tf.range(T):
        h_t = a[:, t, :] * h_t + b[:, t, :]
        h_list = h_list.write(t, h_t)

    h_stack = tf.transpose(h_list.stack(), [1, 0, 2])  # [B, T, H]
    return h_stack

def train_min_gru_parallel(X, Y, vocab_size, hidden_size=10, embedding_dim=8, epochs=500,verbose=True):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    min_gru = MinGRUCell(hidden_size, vocab_size)
    losses = []
    # Embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            h0 = tf.zeros((X.shape[0], hidden_size), dtype=tf.float32)  # Initial hidden state
            embedded_X = embedding_layer(X)  # [B, T, D]

            # Precompute gates
            a, b = min_gru.precompute(embedded_X)  # [B, T, H]

            # Parallel scan to compute h_t over time
            h_all = parallel_scan(a, b, h0)  # [B, T, H]

            # Final prediction from last time step
            y_t = min_gru.output_layer(h_all[:, -1, :])  # [B, vocab_size]

            # Compute categorical crossentropy loss
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(Y, y_t))

        # Update both minGRU and embedding layer
        grads = tape.gradient(loss, min_gru.trainable_variables + embedding_layer.trainable_variables)
        optimizer.apply_gradients(zip(grads, min_gru.trainable_variables + embedding_layer.trainable_variables))
        losses.append(loss.numpy())

        if epoch % 10 == 0 and verbose:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

        if epoch == epochs - 1:
            return min_gru, embedding_layer, losses

def predict_min_gru_parallel(model, tokenizer, input_text, embedding_layer, hidden_size):
    seq_length = 3
    sequence = tokenizer.texts_to_sequences([input_text])[0][-seq_length:]
    X_input = tf.convert_to_tensor([sequence], dtype=tf.int32)
    h0 = tf.zeros((1, hidden_size), dtype=tf.float32)

    embedded_X = embedding_layer(X_input)       # [1, T, D]
    a, b = model.precompute(embedded_X)         # [1, T, H]
    h_all = parallel_scan(a, b, h0)             # [1, T, H]
    y_t = model.output_layer(h_all[:, -1, :])   # [1, vocab_size]

    predicted_index = tf.argmax(y_t, axis=1).numpy()[0]
    return tokenizer.index_word.get(predicted_index, "<UNK>")
