import tensorflow as tf
import numpy as np

# ========== Standard GRU Model ==========

x=100


class StandardGRU(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, x):
        x = self.embedding(x)                         # [B, T, D]
        output_seq, final_state = self.gru(x)         # [B, T, H]
        y = self.dense(output_seq[:, -1, :])          # Predict from final time step
        return y

# ========== Training Function ==========

def train_gru_model(X, Y, vocab_size, embedding_dim=8, hidden_size=10, epochs=100):
    model = StandardGRU(vocab_size, embedding_dim, hidden_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(X)  # Forward pass
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(Y, y_pred))  # Loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 10 == 0:
            print(f"[GRU] Epoch {epoch}, Loss: {loss.numpy():.4f}")

    return model

# ========== Prediction Function ==========

def predict_next_word_gru(model, tokenizer, input_text, seq_length=3):
    sequence = tokenizer.texts_to_sequences([input_text])[0][-seq_length:]
    X_input = tf.convert_to_tensor([sequence], dtype=tf.int32)

    y_pred = model(X_input)
    predicted_index = tf.argmax(y_pred, axis=1).numpy()[0]

    return tokenizer.index_word.get(predicted_index, "<UNK>")
