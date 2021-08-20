import torch
import torchquad
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS, enable_cuda

torch.set_printoptions(precision=10)

def f_1(x):
	return torch.exp(-torch.sum(x[:]**2, dim=1))

def f_2(x):
	return torch.exp(-torch.sum(x[:]**2 + x[:], dim=1))

mc = MonteCarlo()
integration_domain=[]

print("For integration of f(x) = e^(-X.dot(X)) from dimension 1 to 10")

for i in range(1,11):
	print("For dimension = ", i )
	integration_domain.append([-1,1])
	result = mc.integrate(f_1, dim=i, N=1000000, integration_domain = integration_domain)
	print("Integral value = " ,result.item())
	
integration_domain.clear()
print("For integration of f(x) = e^(-X.dot(X) + X.sum()) from dimension 1 to 10")
	
for i in range(1,11):
	print("For dimension = ", i )
	integration_domain.append([-1,1])
	result = mc.integrate(f_2, dim=i, N=1000000, integration_domain = integration_domain)
	print("Integral value = " ,result.item())
