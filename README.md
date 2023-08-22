# NovoLSTM

The `NovoLSTM` is a custom LSTM (Long Short-Term Memory) layer designed with a unique formulation for the input gate, which diverges from traditional LSTM structures.

## 🔑 **Key Features**

### Customized Input Gate

The cornerstone of `NovoLSTM` is its distinct input gate. In traditional LSTMs, the input gate regulates the flow of information into the cell state using a simple sigmoid activation. In contrast, `NovoLSTM` innovates by combining layer normalization, sigmoid activation, and softmax functions to determine this flow.

$$
i_t = \text{LayerNorm}(\sigma(W_i \cdot [h_{t-1}, x_t] + b_i)) \times \text{softmax}(W_{i'} \cdot [h_{t-1}, x_t] + b_{i'})
$$

Where:
- \( \sigma \) is the sigmoid activation.
- \( \text{LayerNorm} \) is layer normalization.
- \( W_i, W_{i'} \) are weight matrices.
- \( b_i, b_{i'} \) are bias vectors.

#### Why This Formulation?

1. **Enhanced Regularization**: Layer normalization can lead to faster convergence and can stabilize the training, especially in deeper networks.

2. **Flexible Information Flow**: By using both sigmoid and softmax activations, the model has a richer way to regulate information flow, potentially allowing it to capture more complex patterns or reduce certain inefficiencies.

3. **Exploratory Dynamics**: Neural network architectures are often enriched through experimentation. This novel approach aims to discover potential improvements in LSTM dynamics and address some of its inherent challenges.

### Other Features

- **Standard LSTM Dynamics**: The forget gate, cell update, and output gate remain consistent with traditional LSTM computations, ensuring the cell retains its core functionalities.
  
- **Layer Normalization**: Apart from the input gate, layer normalization can be expanded to other parts of the LSTM if required.


## 📁 **Code Structure**

- `__init__(self, units)`: Initializes the `NovoLSTM` layer with a specified number of `units`.

- `build(self, input_shape)`: Defines the weight matrices and bias vectors required for the LSTM computations.

- `call(self, inputs, states)`: Contains the main logic of the LSTM computations, which includes the custom input gate dynamics.

## 💡 **Example Usage**

```python
import tensorflow as tf

vocabulary_size = 10000  # size of your vocabulary
embedding_dim = 100      # size of word embeddings
sequence_length = 50     # length of input sequences

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length),
    tf.keras.layers.RNN(NovoLSTM(units=128)),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

