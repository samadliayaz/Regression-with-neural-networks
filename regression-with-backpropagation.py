from math import exp
from random import seed
from random import random


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return max(0, activation)


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def transfer_derivative(output):
    if output < 0:
        return 0
    else:
        return 1


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        l_rate -= 0.0009
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[0] = row[-1]
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs[0]


def main():
    seed(1)
    dataset = []
    minlist = []
    maxlist = []

    for i in range(82):
        minlist.append(100000000)
    for i in range(82):
        maxlist.append(-100000000)

    for i in range(17011):
        temp = input().split("\t")
        for j in range(len(temp)):
            temp[j] = float(temp[j])

        dataset.append(temp)

    for i in range(17011):
        a = float(input())
        dataset[i].append(a)

    for i in range(len(dataset)):
        for j in range(len(dataset[i]) - 1):
            if dataset[i][j] < minlist[j]:
                minlist[j] = dataset[i][j]
            if dataset[i][j] > maxlist[j]:
                maxlist[j] = dataset[i][j]

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = (dataset[i][j] - minlist[j]) / (maxlist[j] - minlist[j])

    testset = []
    for i in range(4252):
        temp = input().split("\t")
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        testset.append(temp)

    for i in range(len(testset)):
        for j in range(len(testset[i])):
            testset[i][j] = (testset[i][j] - minlist[j]) / (maxlist[j] - minlist[j])
            if testset[i][j] > 1:
                testset[i][j] = 1
            elif testset[i][j] < 0:
                testset[i][j] = 0

    n_inputs = len(dataset[0]) - 1
    n_outputs = 1
    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, dataset, 0.1, 100, n_outputs)

    for row in testset:
        prediction = predict(network, row)
        prediction = prediction * (maxlist[-1] - minlist[-1]) + minlist[-1]
        print(prediction, end='\n')


if __name__ == "__main__":
    main()
