<h1 align="center">🧠 Transformer Architectures from Scratch</h1>

<p align="center">
Three Transformer architecture variants — Encoder-Only, Encoder-Decoder, and Decoder-Only — implemented from first principles in TensorFlow/Keras, without relying on any pretrained models.
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
This repository implements the three foundational <b>Transformer architecture families</b> introduced in "Attention Is All You Need," each applied to a different NLP task, and each built from scratch — including manual implementation of multi-head self-attention, positional embeddings, causal masking, and encoder-decoder cross-attention.
</p>

<p>
Most portfolios that mention "Transformers" mean fine-tuning a pretrained model (BERT, GPT-2) via a library like HuggingFace. This repository takes the opposite approach: understanding and re-implementing the underlying mechanics directly, to build a genuine grasp of how these architectures actually work internally.
</p>

<ul>
<li>Understanding self-attention and multi-head attention mechanics</li>
<li>Understanding how bidirectional vs. causal (masked) attention changes what an architecture can do</li>
<li>Understanding encoder-decoder cross-attention and teacher forcing</li>
<li>Understanding autoregressive generation</li>
</ul>

<hr>

<h2>⚙️ Sub-Projects</h2>

<table>
<tr>
<th>Folder</th>
<th>Architecture</th>
<th>Task</th>
</tr>
<tr>
<td><code>Sentiment_Classification_Encoder_Only_Transformer/</code></td>
<td>Encoder-Only (BERT-style)</td>
<td>Binary sentiment classification</td>
</tr>
<tr>
<td><code>English_Marathi_Translation_Encoder-Decoder_Transformer/</code></td>
<td>Encoder-Decoder</td>
<td>English → Marathi neural machine translation</td>
</tr>
<tr>
<td><code>Mini-GPT__Decoder_Only_Transformer/</code></td>
<td>Decoder-Only (GPT-style)</td>
<td>Autoregressive next-word prediction / text generation</td>
</tr>
</table>

<h3>Architecture Comparison</h3>

<table>
<tr>
<th>Architecture</th>
<th>Attention Type</th>
<th>Typical Use Case</th>
<th>Real-World Example</th>
</tr>
<tr>
<td>Encoder-only</td>
<td>Bidirectional self-attention</td>
<td>Understanding / classification</td>
<td>BERT</td>
</tr>
<tr>
<td>Decoder-only</td>
<td>Causal (masked) self-attention</td>
<td>Autoregressive generation</td>
<td>GPT family</td>
</tr>
<tr>
<td>Encoder-Decoder</td>
<td>Bidirectional encoder + causal decoder + cross-attention</td>
<td>Sequence-to-sequence (translation, summarization)</td>
<td>T5, original Transformer</td>
</tr>
</table>

<h3>Implementation Specs at a Glance</h3>

<table>
<tr>
<th></th>
<th>Sentiment (Encoder-Only)</th>
<th>Translation (Encoder-Decoder)</th>
<th>Mini-GPT (Decoder-Only)</th>
</tr>
<tr>
<td>Task</td>
<td>Binary classification</td>
<td>Seq2seq translation</td>
<td>Next-word prediction / generation</td>
</tr>
<tr>
<td>Embedding dim</td>
<td>16</td>
<td>32</td>
<td>32</td>
</tr>
<tr>
<td>Attention heads</td>
<td>2</td>
<td>2</td>
<td>2</td>
</tr>
<tr>
<td>Feed-forward dim</td>
<td>32</td>
<td>64</td>
<td>64</td>
</tr>
<tr>
<td>Sequence length</td>
<td>8</td>
<td>8</td>
<td>6</td>
</tr>
<tr>
<td>Attention type</td>
<td>Bidirectional</td>
<td>Bidirectional (encoder) + Causal (decoder) + Cross-attention</td>
<td>Causal (masked) only</td>
</tr>
<tr>
<td>Output layer</td>
<td>Dense(1, sigmoid)</td>
<td>Dense(vocab_size, softmax)</td>
<td>Dense(vocab_size, softmax)</td>
</tr>
<tr>
<td>Loss function</td>
<td>Binary cross-entropy</td>
<td>Sparse categorical cross-entropy</td>
<td>Sparse categorical cross-entropy</td>
</tr>
</table>

<p><i>Note: verify these values against the final code before relying on them — update any that changed during your revision pass.</i></p>

<hr>

<h2>🛠️ Tech Stack</h2>

<ul>
<li><b>Language:</b> Python</li>
<li><b>Framework:</b> TensorFlow / Keras</li>
</ul>

<h3>Concepts Used</h3>

<ul>
<li>Multi-Head Self-Attention</li>
<li>Token &amp; Positional Embeddings</li>
<li>Causal (Look-Ahead) Masking</li>
<li>Encoder-Decoder Cross-Attention</li>
<li>Teacher Forcing</li>
<li>Autoregressive Generation</li>
<li>Residual Connections &amp; Layer Normalization</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
Transformer_Architectures_from_Scratch/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── Sentiment_Classification_Encoder_Only_Transformer/
│   └── Transformer_Sentiment_Classification_Encoder_Only.py
│
├── English_Marathi_Translation_Encoder-Decoder_Transformer/
│   └── Transformer_Neural_Machine_Translation_Encoder_Decoder.py
│
└── Mini-GPT__Decoder_Only_Transformer/
    └── Transformer_Generative_Pre_trained_Transformer_Decoder_Only.py
</pre>

<hr>

<h2>🚀 Installation</h2>

<h3>1️⃣ Clone the Repository</h3>
<pre>
git clone https://github.com/UmeshBhabad/Transformer_Architectures_from_Scratch.git
cd Transformer_Architectures_from_Scratch
</pre>

<h3>2️⃣ Create a Virtual Environment (Recommended)</h3>
<pre>
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
</pre>

<h3>3️⃣ Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<hr>

<h2>▶️ Usage</h2>

<pre>
# Sentiment classification (encoder-only)
python Sentiment_Classification_Encoder_Only_Transformer/Transformer_Sentiment_Classification_Encoder_Only.py

# English-Marathi translation (encoder-decoder)
python English_Marathi_Translation_Encoder-Decoder_Transformer/Transformer_Neural_Machine_Translation_Encoder_Decoder.py

# Mini-GPT text generation (decoder-only)
python Mini-GPT__Decoder_Only_Transformer/Transformer_Generative_Pre_trained_Transformer_Decoder_Only.py
</pre>

<hr>

<h2>🖥️ How Each Model Works</h2>

<h3>Encoder-Only (Sentiment Classification)</h3>
<ol>
<li>Sentences are tokenized and padded to a fixed length</li>
<li>Token + positional embeddings are combined</li>
<li>A Transformer encoder block applies bidirectional multi-head self-attention, so every word can attend to every other word</li>
<li>Global average pooling collapses the sequence into a single vector</li>
<li>A dense classification head outputs a positive/negative sentiment score</li>
</ol>

<h3>Encoder-Decoder (English-Marathi Translation)</h3>
<ol>
<li>English input is processed by the encoder using bidirectional self-attention</li>
<li>The Marathi target sequence is fed to the decoder using teacher forcing during training</li>
<li>The decoder applies masked self-attention (causal), then cross-attention over the encoder's output</li>
<li>At inference time, translation is generated autoregressively, one word at a time, since no ground-truth target exists yet</li>
</ol>

<h3>Decoder-Only (Mini-GPT)</h3>
<ol>
<li>Input/target pairs are created by shifting a sentence by one token (next-word prediction)</li>
<li>Stacked decoder blocks apply causal (masked) self-attention, so each position can only see previous positions</li>
<li>Given a seed phrase, the model generates text autoregressively — predicting one word, appending it, and repeating</li>
</ol>

<hr>

<h2>🧠 Design Highlights</h2>

<ul>
<li><b>Shared Building Blocks</b> — all three models reuse the same core <code>TokenAndPositionEmbedding</code> pattern, showing how positional encoding underlies every Transformer variant</li>
<li><b>Bidirectional vs. Causal Attention</b> — the encoder-only and decoder-only implementations make the difference between full and masked self-attention concrete, not just theoretical</li>
<li><b>Cross-Attention as the Translation Bridge</b> — the encoder-decoder model explicitly shows how the decoder "looks back" at the source sentence via cross-attention</li>
<li><b>Autoregressive Inference</b> — the GPT-style model implements true token-by-token generation, feeding each predicted word back in as input for the next prediction</li>
</ul>

<hr>

<h2>🔮 Future Enhancements</h2>

<ul>
<li>Scale up training data beyond the current small, purpose-built datasets</li>
<li>Subword tokenization (e.g. Byte-Pair Encoding) instead of whole-word tokenization</li>
<li>Sampling strategies (temperature, top-k, nucleus sampling) instead of greedy decoding for the GPT and translation models</li>
<li>BLEU score evaluation for the translation model</li>
<li>Model checkpointing and saved weights for reuse without retraining</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>

<p>
<b>Umesh Shivaji Bhabad</b><br>
📫 umeshbhabad9@gmail.com
</p>

<hr>

<h2>⭐ Support</h2>

<p>If you find this project useful, consider giving it a ⭐ on GitHub!</p>