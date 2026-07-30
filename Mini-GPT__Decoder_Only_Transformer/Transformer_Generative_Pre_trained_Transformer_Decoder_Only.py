import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# ------------------------------------------------------------
# Utility Header
# ------------------------------------------------------------

def Header(step, title):
    print("\n" + "=" * 75)
    print(f"Step {step}: {title}")
    print("=" * 75)


# ============================================================
# Step 1: Dataset Creation
# ============================================================

Header(1, "Dataset Creation")

print("Creating simple text data for next-word prediction.")

sentences = [
    "i love artificial intelligence",
    "i love machine learning",
    "i love deep learning",
    "machine learning is powerful",
    "deep learning is amazing",
    "artificial intelligence is future",
    "transformer is powerful model",
    "decoder transformer generates text",
    "gpt is decoder only transformer",
    "students learn artificial intelligence",
    "students learn python programming",
    "python is useful language",
    "ai helps students learn",
    "transformer learns word relations",
    "masked attention prevents future words",
    "language model predicts next word",
    "gpt predicts next token",
    "decoder only model generates output",
    "self attention understands context",
    "positional encoding gives word order"
]

for sentence in sentences:
    print(sentence)


# ============================================================
# Step 2: Text Vectorization
# ============================================================

Header(2, "Tokenization and Padding")

print("Converting words into token numbers.")

vocab_size = 1000
sequence_length = 6

vectorizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_sequence_length=sequence_length + 1
)

vectorizer.adapt(sentences)

tokenized_data = vectorizer(sentences)

vocabulary = vectorizer.get_vocabulary()

print("\nVocabulary:")
for index, word in enumerate(vocabulary):
    print(index, ":", word)

print("\nTokenized Sentences:")
for sentence, tokens in zip(sentences[:5], tokenized_data.numpy()[:5]):
    print(sentence)
    print(tokens)
    print("-" * 50)


# ============================================================
# Step 3: Prepare Input and Target
# ============================================================

Header(3, "Input and Target Preparation")

print("""
For Decoder-only Transformer, model learns next-word prediction.

Input  : previous words
Target : next words
""")

x_data = tokenized_data[:, :-1]
y_data = tokenized_data[:, 1:]

print("\nSample Sentence:")
print(sentences[0])

print("\nInput Tokens:")
print(x_data[0].numpy())

print("\nTarget Tokens:")
print(y_data[0].numpy())

print("""
Example:
If sentence is: i love artificial intelligence

Input sequence  : i love artificial
Target sequence : love artificial intelligence
""")


# ============================================================
# Step 4: Token and Positional Embedding
# ============================================================

Header(4, "Token Embedding and Positional Embedding")

print("""
Token Embedding gives meaning to words.
Positional Embedding gives word order information.
""")

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()

        self.token_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embedding = layers.Embedding(
            input_dim=max_len,
            output_dim=embed_dim
        )

    def call(self, x):
        length = tf.shape(x)[-1]

        positions = tf.range(start=0, limit=length, delta=1)

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        return token_embeddings + position_embeddings


# Demonstration of embedding output
embed_dim = 32

sample_embedding_layer = TokenAndPositionEmbedding(
    sequence_length,
    vocab_size,
    embed_dim
)

sample_embedding_output = sample_embedding_layer(x_data[:1])

print("\nSample Input Tokens:")
print(x_data[:1].numpy())

print("\nEmbedding Output Shape:")
print(sample_embedding_output.shape)

print("\nSample Embedding Vector of First Word:")
print(sample_embedding_output[0][0].numpy())