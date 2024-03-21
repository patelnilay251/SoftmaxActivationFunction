import numpy as np

def softmax(x):
    # from each element of the input array
    exp_x = np.exp(x - np.max(x))

    # Computing softmax probabilities
    softmax_probs = exp_x / np.sum(exp_x, axis=0)

    return softmax_probs

    
input_array = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
softmax_output = softmax(input_array)

print("Input array:")
print(input_array)
print("\nSoftmax output:")
print(softmax_output)
