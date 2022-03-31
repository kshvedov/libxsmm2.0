import time
import torch
import random
import pcl_mlp

import numpy as np

torch.manual_seed(77)
random.seed(77)

# MB = 128 * 8
# N = MB
# K=256 #128
# C=256 #64


MB = 64
N = MB
K= 64 #128
C= 64 #64


fc = pcl_mlp.XsmmLinear(C, K)
#fc = pcl_mlp.XsmmLinear(C, K)
tl = torch.nn.Linear(C, K)

sparsity_rate = 0.8
#sparsity_rate = 0.2
weight = torch.zeros(K, C, requires_grad=False)

# Populate weight matrix
for k in range(K):
    for c in range(C):
        # Creating simple permutation matrix
        # This causes segmentation error
        """
        if K-k-1 == c:
            weight[k, c] = 1.
        """
        # This doesn't cause segmentation error
        if random.random() > sparsity_rate:
            weight[k, c] = random.random()
# bias = torch.randn(K, requires_grad=True)
bias = torch.zeros(K, requires_grad=False)
#print("Weight: ", weight)
#print("Bias: ", bias)

"""
x1 = torch.zeros(MB, C, requires_grad=True)
for n in range(N):
    for c in range(C):
        x1[n][c] = (c + n) / 20.
"""

x1 = torch.randn(MB, C, requires_grad=True)
x2 = x1.clone().detach().requires_grad_()

fc.weight.data = weight.clone()
#fc.reset_weight_shape(torch.bfloat16)
tl.weight.data = weight.clone()
fc.bias.data = bias.clone()
tl.bias.data = bias.clone()

###########################################
# Timing
###########################################
time_1 = time.perf_counter()
y1 = fc(x1)
time_1 = time.perf_counter() - time_1
# y2 = tl(x2.to_mkldnn())
time_2 = time.perf_counter() 
y2 = tl(x2)
time_2 = time.perf_counter() - time_2
###########################################
#y2 = y2.to_dense()
z1 = y1.mean()
z2 = y2.mean()

print("xsmm: {}".format(z1))
print("ref: {}".format(z2))

###########################################
# Timing prints
###########################################
print(f"xsmm time: {time_1} s")
print(f" ref time: {time_2} s")
###########################################

if not y1.allclose(y2, rtol=1e-4, atol=1e-4):
    print("forward pass invalid")
    print("ref")
    print(y2)

    print("xsmm")
    print(y1)

###########################################
# Timing
###########################################
#print("██████████ 1 STR ██████████")
time_3 = time.perf_counter()
z1.backward()
time_3 = time.perf_counter() - time_3
#print("██████████ 1 END ██████████")

#print("██████████ 2 STR ██████████")
time_4 = time.perf_counter()
z2.backward()
time_4 = time.perf_counter() - time_4
#print("██████████ 2 END ██████████")
###########################################


# Testing input grad
if not x1.grad.allclose(x2.grad, rtol=1e-6, atol=1e-6):
  print("InputGrad:")
  print(x1.grad.size())
  print("xsmm: ", x1.grad)
  print(x2.grad.size())
  print("ref: ", x2.grad)
  #print((x2.grad-x1.grad).sort(descending=True))
  print(x2.grad-x1.grad)

print("xsmm: {}".format(x1.grad.mean()))
print("ref: {}".format(x2.grad.mean()))

###########################################
# Timing prints
###########################################
print(f"xsmm time: {time_3} s")
print(f" ref time: {time_4} s")
###########################################

print("xsmm", x1.grad)
print(" ref", x2.grad)

f = open("grad_ref_input.txt", "w")
torch.set_printoptions(profile="full")
print(x2.grad, file=f)
torch.set_printoptions(profile="default")
f.close()

# Testing weight grad
#weight_mask = (weight != 0.0).float()
weight_mask = (weight != 0.0).type(torch.float32)
masked_weight_ref = tl.weight.grad * weight_mask

###########################################
# Testing
###########################################

# Alternative --> Mask values
#-----------------------------------------#
# # convert values larger than 10 and smaller than 10-12 to zero
# fc.weight.grad[abs(fc.weight.grad) > 1e9] = 0.
# fc.weight.grad[abs(fc.weight.grad) < 1e-12] = 0.
#-----------------------------------------#

from IPython import embed; embed()


print()
print("█"*100)
print("Checking update kernel")
print("xsmm: {}".format(fc.weight.grad.mean()))
print(" ref: {}".format(masked_weight_ref.mean()))

print(f"Min and Max value of ref matrix: min: {masked_weight_ref.min()}, max: {masked_weight_ref.max()}")
print("█"*100)
print()

# Saving weights to compare
#-----------------------------------------#
np.savetxt('fc.weight.grad.size.txt', fc.weight.grad.numpy())
np.savetxt('tl.weight.grad.size.txt', masked_weight_ref.numpy())
###########################################

if not masked_weight_ref.allclose(fc.weight.grad, rtol=1e-6, atol=1e-6):
    print("WeightGrad:")
    print(fc.weight.grad.size())
    print("xsmm: ", fc.weight.grad)
    print(tl.weight.grad.size())
    print("ref: ", masked_weight_ref)
