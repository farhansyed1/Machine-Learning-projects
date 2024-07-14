import numpy, random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Kernel functions

def linear_kernel(x,y):
    # return (numpy.dot(numpy.transpose(x),y)) 
    return numpy.dot(x,y)

def poly_kernel(x,y): 
    return (numpy.dot(x,y)+1)**10

def rad_bas_func(x,y):
    return math.exp(-numpy.linalg.norm(x-y)**2 / (2 *(5 ** 2)))

def createPMatrix(inputs,targets,dim):
    P=numpy.zeros((dim,dim))
    for i in range(dim): 
        for j in range(dim): 
            P[i,j]= targets[i]*targets[j]*linear_kernel(inputs[i],inputs[j])
    return P


# Generating test data 
# numpy.random.seed(100)

classA= numpy.concatenate((numpy.random.randn(10,2)*0.2+[1.5,0.5],numpy.random.randn(10,2)*0.2+[-1.5,0.5]))
classB = numpy.random.randn(20,2) * 0.2 +[0.0,-0.5]
                                          
inputs = numpy.concatenate((classA,classB))
targets = numpy.concatenate ((numpy.ones(classA.shape[0]) ,-numpy.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows ( samples )
permute=list(range(N))
random.shuffle( permute )
inputs = inputs [permute,:]
targets= targets[permute]


P_matrix = createPMatrix(inputs,targets,N)
tvec = numpy.array(targets)

def objective(vector):
    return 0.5*numpy.sum(numpy.outer(vector,vector)*P_matrix)- numpy.sum(vector)
    # return 0.5*numpy.dot(vector,numpy.dot(vector,P_matrix))- numpy.sum(vector)



def zerofun(vector):
    return numpy.dot(vector,tvec)

C=5
start=numpy.zeros(N)
B=[(0, None) for b in range(N)]
XC={'type':'eq', 'fun':zerofun}

ret = minimize( objective,start, bounds=B, constraints=XC)
alpha = ret['x']

# print(alpha)
print(ret["success"])

data_points=[]
target_values=[]
x_values=[]

# Extract non zero values. 
for i in range(0,len(alpha)): 
    if alpha[i]>10**(-5):
        data_points.append(alpha[i])
        target_values.append(targets[i])
        x_values.append(inputs[i])

# Calculate b value
b=0
s_vec=x_values[0]
for i in range(0,len(data_points)): 
    b+=data_points[i]*target_values[i]*linear_kernel(s_vec,x_values[i])
    
b-=target_values[0]

# Implement indicator
def indicator(sx,sy): 
    s_new_vec=[sx,sy]
    s=0
    for i in range(0,len(data_points)): 
        s+=data_points[i]*target_values[i]*linear_kernel(s_new_vec,x_values[i])
    return s-b


# Plotting class A and B points 
plt.plot([p[0] for p in classA] ,
    [p[1] for p in classA] ,
    "b.")

plt.plot([p[0] for p in classB] ,
    [p[1] for p in classB] ,
    "r.")

plt.plot(x_values, "g+")

plt.axis("equal") # Force same s c a l e on both axes
plt.savefig("svmplot.pdf") # Save a copy i


#Plotting decision boundaries 

xgrid=numpy.linspace(-5,5)
ygrid=numpy.linspace(-4,4)
grid=numpy.array([[indicator(x,y) for x in xgrid ]for y in ygrid ] )
plt.contour(xgrid,ygrid,grid,(-1.0 ,0.0 , 1.0 ), colors =('red' ,'black','blue') ,linewidths =(1 , 3 , 1 ) )
plt.show() 

