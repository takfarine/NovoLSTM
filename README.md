# NovoLSTM

The `NovoLSTM` is a custom LSTM (Long Short-Term Memory) layer designed with a unique formulation for the input gate, which diverges from traditional LSTM structures.

## 🔑 **Key Features**

### Customized Input Gate

The cornerstone of `NovoLSTM` is its distinct input gate. In traditional LSTMs, the input gate regulates the flow of information into the cell state using a simple sigmoid activation. In contrast, `NovoLSTM` innovates by combining layer normalization, sigmoid activation, and softmax functions to determine this flow.

$$
i_t = \text{LayerNorm}(\sigma(W_i \cdot [h_{t-1}, x_t] + b_i)) \times \text{softmax}(W_{i'} \cdot [h_{t-1}, x_t] + b_{i'})
$$

Where:
- $`\sigma`$ is the sigmoid activation.
- $`\text{LayerNorm}`$ is layer normalization.
- $`( W_i, W_{i'})`$ are weight matrices.
- $`( b_i, b_{i'})`$ are bias vectors.

#### Why This Formulation?

1. **Enhanced Regularization**: Layer normalization can lead to faster convergence and can stabilize the training, especially in deeper networks.

2. **Flexible Information Flow**: By using both sigmoid and softmax activations, the model has a richer way to regulate information flow, potentially allowing it to capture more complex patterns or reduce certain inefficiencies.

3. **Exploratory Dynamics**: Neural network architectures are often enriched through experimentation. This novel approach aims to discover potential improvements in LSTM dynamics and address some of its inherent challenges.


## Mathematical Rationale Behind the Custom Input Gate

Traditional LSTM units utilize the sigmoid function, denoted by $`\sigma`$, for the input gate, which regulates the flow of information into the cell state. This ensures that values lie between 0 and 1, representing how much of the new information should flow into the cell state.

Your custom LSTM, `NovoLSTM`, uses a unique combination of sigmoid activation, layer normalization, and softmax.

### 1. Sigmoid Activation:

Given by 
$`[ \sigma(z) = \frac{1}{1 + e^{-z}} ]`$,
this function squashes its input to produce an output between 0 and 1. In the context of LSTM, values closer to 1 allow more information to pass through, while values closer to 0 inhibit the flow of information.

### 2. Layer Normalization:

Layer normalization is used to stabilize the activations in the network. It is defined as:

$`[ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} ]`$

where:
- $`( x )`$ is the input vector.
- $`( \mu )`$ is the mean of the input vector.
- $`( \sigma^2 )`$ is the variance of the input vector.
- $`( \epsilon )`$ is a small number to prevent division by zero.

The normalized data $`( \hat{x} )`$  is then scaled and shifted by learnable parameters. Layer normalization's primary benefit is that it can lead to faster convergence and reduce the sensitivity to the initial weights.

### 3. Softmax Activation:

This function exponentiates its input and then normalizes it. It is typically used in the output layer of a classification network because its output can be interpreted as probabilities. In the context of your input gate, the softmax ensures that the input values are normalized to a distribution of values that sum to one. This might be useful to prioritize certain features over others when updating the cell state.

The combined equation for the input gate in `NovoLSTM` is:

\[ i_t = \text{LayerNorm}(\sigma(W_i \cdot [h_{t-1}, x_t] + b_i)) \times \text{softmax}(W_{i'} \cdot [h_{t-1}, x_t] + b_{i'}) \]

The rationale for this combined approach is:
- The sigmoid function provides the gating mechanism, deciding which values are allowed to flow.
- Layer normalization stabilizes these values, ensuring that no particular feature overwhelms the others due to scale differences.
- The softmax, when multiplied with the sigmoid output, offers a way of prioritizing or giving attention to certain features more than others.

By this formulation, the network can potentially learn which features to prioritize in different contexts, making it dynamic and more adaptive to intricate data patterns. However, as with many neural network enhancements, the effectiveness of this method is highly dependent on the specific task and the data at hand. Regular evaluations and comparisons with traditional LSTM structures are advisable to ensure that this new formulation provides tangible benefits.



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

