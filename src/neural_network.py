import numpy as np

class Sample:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return ",".join(map(str, self.x)) + ", label = " + str(self.y)

def read_data(dataFileName):
    import os
    assert(os.path.exists(dataFileName))
    lines = []
    samples = []
    ifile = open(dataFileName, "r")
    for (index, string) in enumerate(ifile):
        if (index == 0):
            header = string.strip("\n")
        else:
            a = map(int, string.strip("\n").split(","))
            y = a[0]
            assert(y >= 0 and y <= 9)
            x = np.asarray(map(lambda ele: ele/255.0, a[1:]))
            samples.append(Sample(x, y))
    ifile.close()
    return samples

def Z(net):
    result = 1.0
    for i in range(len(net)):
        result += np.exp(net[i])
    return result

def Loss(net, sample):
    label = sample.y
    if (label == 0):
        delta = 0
    else:
        delta = net[label-1]
    result = np.log(Z(net)) - delta
    return result

def Loss_total(net_inputs, samples):
    result = 0.0
    assert(len(net_inputs) == len(samples))
    for i in range(len(net_inputs)):
        result += Loss(net_inputs[i], samples[i])
    return result/float(len(samples))

def linear_transform(W, b, x):
    (row, col) = W.shape
    assert(col == len(x))
    return W.dot(x) + b

def initialize_parameters(row, col):
    import random
    W = np.zeros((row, col))
    b = np.zeros(row)
    for i in range(row):
        for j in range(col):
            W[i, j] = random.uniform(-1, 1)
        b[i] = random.uniform(-1, 1)
    return W, b

def forward_propagation(sample, weight_matrices, biases, activation_function):
    assert(len(weight_matrices) == len(biases))
    x = sample.x
    y = sample.y
    for i in range(len(weight_matrices) - 1):
        W = weight_matrices[i]
        b = biases[i]
        net = linear_transform(W, b, x)
        output_vector = map(activation_function, net)
        x = output_vector
    net = linear_transform(weight_matrices[-1], biases[-1], x)
    return Loss(net, sample)

def forward_propagation_total(samples, weight_matrices, biases, activation_function):
    result = 0.0
    for sample in samples:
        result += forward_propagation(sample, weight_matrices, biases, activation_function)
    return result/float(len(samples))

def relu(x):
    if (x < 0):
        return 0
    else:
        return x

def relu_prime(x):
    if (x < 0):
        return 0.0
    else:
        return 1.0

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_prime(x):
    y = sigmoid(x)
    return y*(1.0 - y)

def flatten(weight_matrix):
    (row, col) = weight_matrix.shape
    result = []
    for i in range(row):
        for j in range(col):
            result.append(weight_matrix[i, j])
    return result

def nablaE(net, sample):
    y = sample.y
    assert(y >= 0 and y <= len(net))
    if (y == 0):
        delta = 0
    result = np.zeros(len(net))
    for i in range(len(net)):
        if (i+1 == y):
            delta = 1
        else:
            delta = 0
        result[i] = np.exp(net[i])/(1 + sum(map(np.exp, net))) - delta
    return result

def dyadic_product(a, b):
    a = a.reshape((len(a), 1))
    b = b.reshape((1, len(b)))
    return a.dot(b)

def matrix_vector_element_wise_product(W, v):
    (row, col) = W.shape
    assert(col == len(v))
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            result[i, j] = W[i, j] * v[j]
    return result

def gradient(weight_matrices, biases, sample, activation_function, activation_function_prime):
    assert(len(weight_matrices) == len(biases))
    W_0 = weight_matrices[0]
    W_1 = weight_matrices[1]
    b_0 = biases[0]
    b_1 = biases[1]
    x_0 = sample.x
    net_0 = linear_transform(W_0, b_0, x_0)
    x_1 = np.asarray(map(activation_function, net_0))
    net_1 = linear_transform(W_1, b_1, x_1)
    nabla = nablaE(net_1, sample)
    nabla = nabla.reshape((len(nabla), 1))
    result = []
    result.append(dyadic_product(nabla, x_1))
    result.append(nabla)
    temp = nabla.transpose().dot(matrix_vector_element_wise_product(W_1, map(activation_function_prime, net_0)))
    temp = temp.transpose()
    result.append(dyadic_product(temp, x_0))
    result.append(temp)
    return result

def gradient_total(weight_matrices, biases, samples, activation_function, activation_function_prime):
    result = []
    for i in range(len(samples)):
        sample = samples[i]
        g = gradient(weight_matrices, biases, sample, activation_function, activation_function_prime)
        if (i == 0):
            for j in range(len(g)):
                result.append(g[j])
        else:
            for j in range(len(g)):
                result[j] = result[j] + g[j]
    for i in range(len(result)):
        result[i] = result[i]/float(len(samples))
    return result

def gradient_norm(g):
    result = 0.0
    for i in range(len(g)):
        result += np.linalg.norm(g[i])
    return result

def gradient_descent(eta, samples, weight_matrices, biases, activation_function, activation_function_prime):
    counter = 0
    iterationMax = 10
    errorLimit = 1.0e-3
    while(counter < iterationMax):
        counter += 1
        g = gradient_total(weight_matrices, biases, samples, activation_function, activation_function_prime)
        weight_matrices[1] = weight_matrices[1] - eta*g[0]
        biases[1] = biases[1] - eta*g[1].transpose()[0]
        weight_matrices[0] = weight_matrices[0] - eta*g[2]
        biases[0] = biases[0] - eta*g[3].transpose()[0]
        print "Counter = ", counter, ", gradient norm = ", gradient_norm(g)
    return weight_matrices, biases

def add_list(a, b):
    assert(len(a) == len(b))
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c

def adam(eta, samples, weight_matrices, biases, activation_function, activation_function_prime):
    assert(eta > 0)
    beta_1 = 0.9
    beta_2 = 0.99
    assert(beta_1 > 0 and beta_1 < 1)
    assert(beta_2 > 0 and beta_2 < 1)
    counter = 0
    iterationMax = 200
    eps = 1.0e-8
    errorLimit = 1.0e-3
    m = []
    v = []
    while(counter < iterationMax):
        counter += 1
        g = gradient_total(weight_matrices, biases, samples, activation_function, activation_function_prime)
        if (counter == 1):
            m = map(lambda ele: (1 - beta_1)*ele, g)
            v = map(lambda ele: (1 - beta_2)*ele**2, g)
        else:
            m = add_list(map(lambda ele: beta_1*ele, m), map(lambda ele: (1 - beta_1)*ele, g))
            v = add_list(map(lambda ele: beta_2*ele, v), map(lambda ele: (1 - beta_2)*ele**2, g))
        weight_matrices[1] = weight_matrices[1] - eta*m[0]/(np.sqrt(v[0]) + eps)
        biases[1] = biases[1] - eta*(m[1]/(np.sqrt(v[1]) + eps)).transpose()[0]
        weight_matrices[0] = weight_matrices[0] - eta*m[2]/(eps + np.sqrt(v[2]))
        biases[0] = biases[0] - eta*(m[3]/(eps + np.sqrt(v[3]))).transpose()[0]
        print "Counter = ", counter, ", gradient norm = ", gradient_norm(g)
    return weight_matrices, biases

def gradient_check(samples, weight_matrices, biases, activation_function):
    E = forward_propagation_total(samples, weight_matrices, biases, activation_function)
    eps = 1.0e-6
    db_1 = []
    db_0 = []
    for i in range(len(biases[0])):
        biases[0][i] += eps
        delta_E = forward_propagation_total(samples, weight_matrices, biases, activation_function) - E
        db_0.append(delta_E/eps)
        biases[0][i] -= eps
    for i in range(len(biases[1])):
        biases[1][i] += eps
        delta_E = forward_propagation_total(samples, weight_matrices, biases, activation_function) - E
        db_1.append(delta_E/eps)
        biases[1][i] -= eps
    print "db_1: ", db_1
    print "db_0: ", db_0
    return np.asarray(db_0), np.asarray(db_1)

def matrix_to_string(matrix):
    result = "\n".join(map(lambda row: ",".join(map(str, row)), matrix))
    return result

def save_model(weight_matrices, biases, modelFileName):
    assert(len(weight_matrices) == len(biases))
    ofile = open(modelFileName, "w")
    ofile.write("Number of weight matrices = " + str(len(weight_matrices)) + "\n")
    for i in range(len(weight_matrices)):
        ofile.write("W_" + str(i) + " matrix row number = " + str(len(weight_matrices[i])) + ", column number = "  + str(len(weight_matrices[i][0])) + "\n")
        ofile.write(matrix_to_string(weight_matrices[i]) + "\n")
        ofile.write("b_" + str(i) + ":\n")
        ofile.write(",".join(map(str, biases[i])) + "\n")
    ofile.close()

def read_model(model_file_name):
    import os
    assert(os.path.exists(model_file_name))
    weight_matrices = []
    biases = []
    lines = []
    ifile = open(model_file_name, "r")
    for (index, string) in enumerate(ifile):
        lines.append(string.strip("\n"))
    ifile.close()
    for i in range(len(lines)):
        if ("matrix" in lines[i]):
            a = lines[i].split(",")
            row_number = int(a[0].split(" = ")[1])
            col_number = int(a[1].split(" = ")[1])
            weight_matrix = np.zeros((row_number, col_number))
            for j in range(i + 1, i + row_number):
                weight_matrix[j - i - 1] = np.asarray(map(float, lines[j].split(",")))
            weight_matrices.append(weight_matrix)
            i = j+1
        elif("b_" in lines[i]):
            bias = np.asarray(map(float, lines[i+1].split(",")))
            biases.append(bias)
            i = i+2
    assert(len(weight_matrices) == len(biases))
    return weight_matrices, biases

def train_model(trainFileName, eta, activation_function, activation_function_prime):
    samples = read_data(trainFileName)
    feature_length = len(samples[0].x)
    weight_matrices = []
    biases = []
    layer_number = 2
    W, b = initialize_parameters(100, feature_length)
    weight_matrices.append(W)
    biases.append(b)
    W, b = initialize_parameters(9, 100)
    weight_matrices.append(W)
    biases.append(b)
    #weight_matrices, biases = gradient_descent(eta, samples, weight_matrices, biases, activation_function, activation_function_prime)
    weight_matrices, biases = adam(eta, samples, weight_matrices, biases, activation_function, activation_function_prime)
    save_model(weight_matrices, biases, "model_parameters.txt")
    return weight_matrices, biases

def probability(sample, weight_matrices, biases, activation_function, category_number):
    assert(len(weight_matrices) == len(biases))
    input_vector = sample.x
    for i in range(len(weight_matrices) - 1):
        net = linear_transform(weight_matrices[i], biases[i], input_vector)
        output_vector = map(activation_function, net)
        input_vector = output_vector
    net = linear_transform(weight_matrices[-1], biases[-1], input_vector)
    assert(category_number == len(net) + 1)
    partition = Z(net)
    p = []
    for i in range(category_number):
        if (i == 0):
            p.append(1.0/partition)
        else:
            p.append(np.exp(net[i-1])/partition)
    return p

def cross_validation(trainFileName, testFileName):
    eta = 1.0e-3
    activation_function = relu
    activation_function_prime = relu_prime
    print "Beginning to train the model ... "
    weight_matrices, biases = train_model(trainFileName, eta, activation_function, activation_function_prime)
    print "Model training finished. Testing the model ..."
    test_samples = read_data(testFileName)
    label_set = set(map(lambda sample: sample.y, test_samples))
    category_number = len(label_set)
    correct_count = 0
    ofile = open("label_prediction.txt", "w")
    for i in range(len(test_samples)):
        sample = test_samples[i]
        label = sample.y
        p = probability(sample, weight_matrices, biases, activation_function, category_number)
        prediction = np.argmax(p)
        ofile.write(str(label) + "," + str(prediction) + "\n")
        if (label == prediction):
            correct_count += 1
    accuracy = float(correct_count)/float(len(test_samples))
    ofile.write("Accuracy = " + str(accuracy) + "\n")
    ofile.close()
    print "Done. "

def main():
    import sys
    eta = 1.0e-3
    cross_validation("train.csv", "test.csv")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
