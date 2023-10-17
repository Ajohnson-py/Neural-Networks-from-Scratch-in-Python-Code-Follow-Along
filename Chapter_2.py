# libraries needed:
import numpy as np

# model of a single neuron with 3 inputs:
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = (inputs[0] * weights[0] +
          inputs[1] * weights[1] +
          inputs[2] * weights[2] + bias)

print(output)
print("-------------")

# model of a single neuron with 4 inputs:
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

output = (inputs[0] * weights[0] +
          inputs[1] * weights[1] +
          inputs[2] * weights[2] +
          inputs[3] * weights[3] + bias)

print(output)
print("-------------")

# model of a layer of 3 neuron with 4 inputs:
inputs = [1, 2, 3, 2.5]
# each neuron in the layer must have a weight
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
# each neuron in the layer must have a bias
bias1 = 2
bias2 = 3
bias3 = 0.5

# since we have 3 neurons, there will be 3 outputs,
# but we will use a list so all outputs are assigned to 1 variable
output = [
          # Neuron 1:
          inputs[0] * weights1[0] +
          inputs[1] * weights1[1] +
          inputs[2] * weights1[2] +
          inputs[3] * weights1[3] + bias1,

          # Neuron 2:
          inputs[0] * weights2[0] +
          inputs[1] * weights2[1] +
          inputs[2] * weights2[2] +
          inputs[3] * weights2[3] + bias2,

          # Neuron 3:
          inputs[0] * weights3[0] +
          inputs[1] * weights3[1] +
          inputs[2] * weights3[2] +
          inputs[3] * weights3[3] + bias3
]

print(output)
print("-------------")

# a more dynamic approach to model a layer with 3 neurons, where each has 4 inputs:
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
# output of neurons in layer
layer_outputs = []

# loops through the neurons and biases
for neuron_weights, neuron_bias in zip(weights, biases):
    # temporary place to store the output of a neuron
    neuron_output = 0
    # loops through the input values and weights
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight

    # adds bias to output
    neuron_output += neuron_bias
    # appends output to layer_outputs
    layer_outputs.append(neuron_output)

print(layer_outputs)
print("-------------")

# model of a single neuron using numpy:
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

outputs = np.dot(weights, inputs) + bias

print(outputs)
print("-------------")

# model of a layer of 3 neuron with 4 inputs using numpy:
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
# in this case, order matters for np.dot() because weights is a matrix
layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)
print("-------------")

# method to transpose a matrix
a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T # the .T is what makes the vector a colum vector

print(np.dot(a, b))
print("-------------")

# model of a layer of neurons with a batch of inputs
inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)
