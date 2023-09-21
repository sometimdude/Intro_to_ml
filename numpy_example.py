import numpy as np 

data = np.array([1,2,3,4,5])
weights = np.random.rand(5,1)

print(data.shape)
print(data.dtype) 

print(weights.shape)
print(weights.dtype)

print(weights.T)
print(weights[0])
print(weights[1])
print(weights[1:3])
print(weights[-1])

def weight_summary(weight_arry):
    for i in range(len(weight_arry)):
        print("Weight`Index`", i, ":", weight_arry[i])
        weight_summary(weights)


noise_magnitude = 0.1 * np.mean(data)

noise = np.random.normal(0, noise_magnitude, data.shape)

noisy_data = data + noise_magnitude * noise

print("Original`data:" , data)
print("Noisy`data" , noisy_data)

negative_noise_mask = data > noisy_data
negative_data = data[negative_noise_mask]
neg_magnitude = np.linalg.norm(negative_data, ord=2)


print("Negative`noise`condition:" , negative_noise_mask)
print("Data`with`negative`noise:" , negative_data)
print(f"Magnitude`of`negative`noise:`{neg_magnitude}")

weights.transposed = weights.T
output = np.dot(weights_transport, data)

print("Data`shape:", data.shape)
print("Transposed`weights`shape:" , weights_transposed.shape)
print("Output`shape:", output.shape)

data = data.reshape(data.shape[0], 1)
output = np.dot(weights_transposed, data)

print("Data`shape:", data.shape)
print(" Output`shape:", output.shape)

data = data.reshape(data.shape[0], 1)
output = np.dot(weights_transposed, data)

print("New`data`shape:", data.shape)
print(" Output`shape:", output.shape)

data_centered = data - np.mean(data, axis=0)


cov_matrix = np.cov(data_centered - np.mean(data, axis=0), rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indicies = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indicies]
sorted_eigenvectors = eigenvectors[:, sorted_indicies]

k=2
top_k_eigenvectors = sorted_eigenvalues[:, :k]

principal_components = np.dot(data_centered, top_k_eigenvectors)
                              

                              
                              
print("Principal`components`shape:", principal_components.shape)


