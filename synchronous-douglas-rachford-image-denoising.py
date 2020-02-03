from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np

# Image path
file_path = 'data/img04g.tif'

im = Image.open(file_path)
imarray = np.array(im)

imarray = imarray[:200, :200]
print("The size of the input figure is: ", imarray.shape)

# Parameters

# image size
imgHeight = imarray.shape[0]
imgWidth = imarray.shape[1]

# display size
sHeight = imgHeight/40
sWidth = imgWidth/40

# gaussian white noise
nMean = 0
nStd = 8
nVariance = nStd ** 2

# gaussian noncausal prediction filter
g_filter = np.asarray([
    [1/12, 1/6, 1/12],
    [1/6, 0, 1/6],
    [1/12, 1/6, 1/12],
])

# other parameters
rho = 100
alpha_param = 0.5

plt.figure(figsize = (sHeight, sWidth))
plt.imshow(imarray, cmap=plt.get_cmap('gray'))

# Add Gaussian Noise.
noisy_img = imarray + (np.random.randn(imarray.shape[0], imarray.shape[1]) * nStd + nMean)
plt.figure(figsize = (sHeight, sWidth))
plt.imshow(noisy_img, cmap=plt.get_cmap('gray'))


def calculate_prediction_variance(x, g_filter, p=2):
  """
  x: 2d image
  g_filter: 3*3 filter
  """

  sigmaP_sum = []
  for i in range(1, x.shape[0] - 1):
    for j in range(1, x.shape[1] - 1):

      x_center = x[i, j]
      for pi in range(3):
        for pj in range(3):
          x_neighbor = x[i + pi - 1, i + pj - 1]
          sigmaP_sum.append(g_filter[pi, pj] * np.abs(x_center - x_neighbor) ** p)

  return np.mean(sigmaP_sum)

variance_x = calculate_prediction_variance(noisy_img, g_filter)


def calculate_bar(Z):
  """
  Z: height * width * 3 * 3 matrix
  """

  height = Z.shape[0]
  width = Z.shape[1]
  z_consensus = np.zeros((height, width))
  for vi in range(height):
    for vj in range(width):
      v_sum = 0
      for pi in range(3):
        for pj in range(3):
          v_sum += Z[
            (vi + pi - 1 + height) % height, (vj + pj - 1 + width) % width,
            2 - pi, 2 - pj
          ]
      z_consensus[vi, vj] = v_sum / 9

  return z_consensus


def consensus_project_voxel(Z):
  """
  Z: height * width * 3 * 3 matrix
  """

  height = Z.shape[0]
  width = Z.shape[1]

  x = np.zeros(Z.shape)
  z_consensus = calculate_bar(Z)

  for vi in range(height):
    for vj in range(width):
      for pi in range(3):
        for pj in range(3):
          x[vi, vj, pi, pj] = z_consensus[
            (vi + pi - 1 + height) % height,
            (vj + pj - 1 + width) % width
          ]

  return x


# Calculate some coefficients

alpha = np.sum(g_filter ** 2 / (g_filter + variance_x / rho))
alpha = alpha / variance_x

gamma = 1 / variance_x + 1 / nVariance + 1 / rho

beta = g_filter / (g_filter * rho + variance_x)
beta = np.reshape(beta, (beta.shape[0] * beta.shape[1], 1))  # row first

phi = variance_x / (g_filter * rho + variance_x)
phi = np.reshape(phi, (phi.shape[0] * phi.shape[1], 1))  # row first


print(" ****** PARAMETERS ****** ")
print("alpha = {}".format(alpha))
print("gamma = {}".format(gamma))
print("beta = {}".format(beta))
print("phi = {}".format(phi))

# initialize z

z = np.random.randint(0, 255, (imgHeight, imgWidth, g_filter.shape[0], g_filter.shape[1]))
# z = np.repeat(noisy_img[:, :, np.newaxis], 3, axis=-1)
# z = np.repeat(z[:, :, :, np.newaxis], 3, axis=-1)
x_hat = np.zeros((imgHeight, imgWidth, g_filter.shape[0], g_filter.shape[1]))

z_entries_num = imgHeight * imgWidth * g_filter.shape[0] * g_filter.shape[1]

epoch_num = 50

for epoch_i in range(epoch_num):

  x_hat = consensus_project_voxel(z)

  x_hat[x_hat > 255] = 255
  x_hat[x_hat < 0] = 0

  z_memory = z.copy()

  if not epoch_i % 10:
    x_display = x_hat[:, :, 1, 1]
    plt.figure(figsize=(sHeight, sWidth))
    plt.imshow(x_display, cmap=plt.get_cmap('gray'))
    plt.show()

  for pi in range(imgHeight):
    for pj in range(imgWidth):
      z_i = z[pi, pj, :, :]
      prox_point = 2 * x_hat[pi, pj] - z_i
      prox_point_vec = np.reshape(prox_point, (prox_point.shape[0] * prox_point.shape[1], 1))

      x_i = 1 / (gamma - alpha) * (1 / nVariance * noisy_img[pi, pj] +
                                   1 / rho * prox_point_vec[4] + np.matmul(beta.T, prox_point_vec))
      x_j_vec = rho * x_i * beta + phi * prox_point_vec
      x_vec = np.reshape(x_j_vec, (prox_point.shape[0], prox_point.shape[1]))
      x_vec[1, 1] = x_i

      z_update = z_i + 2 * alpha_param * (x_vec - x_hat[pi, pj])

      z[pi, pj] = z_update

  print("Epoch {}, the difference between z (square sum): {}".format(
    epoch_i,
    (np.sum((z - z_memory) ** 2))
  )
  )
  print("Epoch {}, the difference between z (averaged value): {}".format(
    epoch_i,
    (np.sum((z - z_memory) ** 2)) ** 0.5 / z_entries_num
  )
  )

x_display = x_hat[:, :, 1, 1]
plt.figure(figsize = (sHeight, sWidth))
plt.imshow(x_display, cmap=plt.get_cmap('gray'))
plt.show()
