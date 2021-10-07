import torch
import torchquad
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS, enable_cuda

torch.set_printoptions(precision=10)

# Enable GPU support if available
# enable_cuda()

def f_1(x):
	return torch.exp(-torch.sum(x[:]**2, dim=1))

def f_2(x):
	return torch.exp(-torch.sum(x[:]**2 + x[:], dim=1))

mc = MonteCarlo()
integration_domain=[]

print("For integration of f(x) = e^(-X.dot(X)) from dimension 1 to 100")
value1 = 1.493648
value2 = 1.691664

for i in range(1,101):
	print("For dimension = ", i )
	integration_domain.append([-1,1])
	result = mc.integrate(f_1, dim=i, N=100000, integration_domain = integration_domain)
	print("By multiplying = ", value1)
	value1 *= 1.493648
	print("Integral value = " ,result.item())
	
integration_domain.clear()
print("For integration of f(x) = e^(-X.dot(X) + X.sum()) from dimension 1 to 100")
	
for i in range(1,101):
	print("For dimension = ", i )
	integration_domain.append([-1,1])
	result = mc.integrate(f_2, dim=i, N=100000, integration_domain = integration_domain)
	print("By multiplying = ", value2)
	value2 *= 1.691664
	print("Integral value = " ,result.item())

