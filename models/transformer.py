import tensorflow as tf
import numpy as np

# ========== Transformer Block ==========

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


# ========== Transformer Model ==========

class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, seq_length, embed_dim=32, num_heads=2, ff_dim=64):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=seq_length)
        # self.pos_encoding = self.add_weight("pos_encoding", shape=[1, seq_length, embed_dim],
        #                                     initializer="random_normal", trainable=True)
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=[1, seq_length, embed_dim],
            initializer="random_normal",
            trainable=True
        )
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, x, training=False):
        x = self.embedding(x)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.transformer_block(x, training=training)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)


# ========== Training Function ==========

def train_transformer_model(X, Y, vocab_size, seq_length=3, embed_dim=32, num_heads=2,
                            ff_dim=64, epochs=500, verbose=True):
    model = TransformerModel(vocab_size, seq_length, embed_dim, num_heads, ff_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    losses = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(X, training=True)
            loss = tf.reduce_mean(loss_fn(Y, y_pred))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(loss.numpy())

        if epoch % 10 == 0 and verbose:
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, axis=1), tf.argmax(y_pred, axis=1)), tf.float32))
            print(f"[Transformer] Epoch {epoch}, Loss: {loss.numpy():.4f}, Accuracy: {acc.numpy():.4f}")

    return model,losses


# ========== Prediction Function ==========

def predict_next_word_transformer(model, tokenizer, input_text, seq_length=3):
    sequence = tokenizer.texts_to_sequences([input_text])[0][-seq_length:]
    X_input = tf.convert_to_tensor([sequence], dtype=tf.int32)
    y_pred = model(X_input, training=False)
    predicted_index = tf.argmax(y_pred, axis=1).numpy()[0]
    return tokenizer.index_word.get(predicted_index, "<UNK>")
