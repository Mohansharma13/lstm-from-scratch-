import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        # apply sin to even indices in the array; cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class SimpleTransformer(tf.keras.Model):
    def __init__(self, vocab_size, max_len=100, embed_dim=64, num_heads=2, ff_dim=128):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_encoding = PositionalEncoding(max_len, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, x, training=False):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_block(x, training=training)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

def train_transformer(X, Y, vocab_size, embed_dim=64, num_heads=2, ff_dim=128, epochs=500, verbose=True):
    model = SimpleTransformer(vocab_size, max_len=X.shape[1], embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    losses = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(X, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(Y, y_pred))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(loss.numpy())

        if epoch % 10 == 0 and verbose:
            print(f"[Transformer] Epoch {epoch}, Loss: {loss.numpy():.4f}")

    return model, losses

def predict_next_word_transformer(model, tokenizer, input_text, seq_length=3):
    sequence = tokenizer.texts_to_sequences([input_text])[0][-seq_length:]
    X_input = tf.convert_to_tensor([sequence], dtype=tf.int32)

    y_pred = model(X_input, training=False)
    predicted_index = tf.argmax(y_pred, axis=1).numpy()[0]

    return tokenizer.index_word.get(predicted_index, "<UNK>")
