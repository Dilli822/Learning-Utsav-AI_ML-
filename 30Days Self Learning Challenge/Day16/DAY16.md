1. Gaussian Filtering

Gaussian Function:
G(x, y) = (1 / (2πσ²)) * e^(-(x² + y²) / (2σ²))

Convolution Operation:
S(t) = ∫ E(x) * G(t - x) dx

2. Sobel Edge Detection

1D Sobel Kernel:
K = [-1, 0, 1]

2D Sobel Kernels:
G_x = [[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]

G_y = [[1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]]

Convolution Operation:
E(t) = ∑ E(k) * K(t-k) for k = -1 to 1
