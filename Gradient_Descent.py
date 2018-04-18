import math

#*******************FIRST FUNCTION*******************
x = 4

#FUNCTION f(x) = x^4 - 3 x^3
f = lambda x: x**4 - 3 * x**3

#DERIVATIVE f'(x) = 4 x^3 - 9 x^2
df = lambda x: 4 * x**3 - 9 * x**2

#ITERATE
for i in range(1000):
	#GRADIENT DESCENT
	x = x - 0.01 * df(x)

print(x)


#*******************SECOND FUNCTION******************
x = 4

f = lambda x: 3*x - math.log(x)
df = lambda x: 3 - (1./x)

for i in range(1000):
	x = x - 0.01 * df(x)

print(x)