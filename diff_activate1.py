import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def srelu(x):
    return np.maximum(-1, x)  

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# Generate data
x = np.linspace(-10, 2, 1000)

# Apply activation functions
relu_result = relu(x)
leaky_relu_result = leaky_relu(x)
srelu_result = srelu(x)
elu_result = elu(x)

import scienceplots
with plt.style.context('science'):
    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.plot(x, relu_result, label="ReLU", color='#7A506D', linewidth=3)
    plt.plot(x, leaky_relu_result, label=r"Leaky ReLU ($\alpha=0.1$)", color='#228B22', linewidth=3)
    plt.plot(x, srelu_result, label="Shifted ReLU ($f(x) = \max(-1, x)$)", color='#B74A33', linewidth=3)
    plt.plot(x, elu_result, label=r"ELU ($\alpha=1.0$)", color='#87CEEB', linewidth=3)

    # Customize the plot
    plt.title("ReLU, Leaky ReLU, Shifted ReLU, and ELU",fontsize=24)
    plt.xlabel("x",fontsize=22)
    plt.ylabel("f(x)",fontsize=22)
    
    plt.legend(loc="best",fontsize=22)
    plt.grid(True)

    # Set x and y axis limits
    plt.xlim(-10, 2)
    plt.ylim(min(np.min(relu_result), np.min(leaky_relu_result), np.min(srelu_result), np.min(elu_result)),
             max(np.max(relu_result), np.max(leaky_relu_result), np.max(srelu_result), np.max(elu_result)))
    plt.tick_params(axis='both', which='major', labelsize=20)  # 设置坐标轴数字的字体大小

    # Hide the x and y axes
    plt.axis('on')

    # Save the figure
    plt.savefig("data/activate.png", dpi=300)

    # Display the plot
    plt.show()
