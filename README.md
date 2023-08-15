# NovoLSTM

The `NovoLSTM` is a custom LSTM (Long Short-Term Memory) layer designed to incorporate specialized dynamics in its input gate. By employing a combination of layer normalization, sigmoid activation, and softmax functions on the input gate, this modified LSTM may offer distinct learning dynamics compared to traditional LSTMs.

## 🔑 **Key Features**

1. **Customized Input Gate**: 
    - The input gate uses a unique formulation:
     ![Equation](https://latex.codecogs.com/svg.image?&space;i_t=\text{LayerNorm}(\sigma(W_i\cdot[h_{t-1},x_t]&plus;b_i))\times\text{softmax}(W_{i'}\cdot[h_{t-1},x_t]&plus;b_{i'}))

    Where:
        - \( \sigma \) is the sigmoid activation.
        - \( \text{LayerNorm} \) is layer normalization.
        - \( W_i, W_{i'} \) are weight matrices.
        - \( b_i, b_{i'} \) are bias vectors.

2. **Standard LSTM Dynamics**: 
    - The forget gate, cell update, and output gate remain consistent with traditional LSTM computations.

3. **Layer Normalization**: 
    - Applied on the input gate to potentially stabilize learning by normalizing the activations.

## 📁 **Code Structure**

- `__init__(self, units)`: Initializes the `NovoLSTM` layer with a specified number of `units`.

- `build(self, input_shape)`: Defines the weight matrices and bias vectors required for the LSTM computations.

- `call(self, inputs, states)`: Contains the main logic of the LSTM computations, which includes the custom input gate dynamics.

## 💡 **Example Usage**

```python
import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
    NovoLSTM(units=128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam')
