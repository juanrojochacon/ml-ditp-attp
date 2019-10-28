###############################################################################################
# In this example we walk through the polynomial regression examples
# that we have seen in the lectures
# Original version from arXiv:1803.08823, Phys.Rept. 810 (2019) 1-124
# "A high-bias, low-variance introduction to Machine Learning for physicists"
# P. Mehta, M. Bukov, C.-H. Wang, A. G. R. Day, C. Richardson, C. K. Fisher, D. J. Schwab.
# Adapted by Juan Rojo, j.rojo@vu.nl
###############################################################################################

#########################################################
# Import the relevant libraries
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt, rcParams
########################################################

# figure where results will be output
fig = plt.figure(figsize=(8, 6))

# Define the properties of the training dataset
N_train=100  # Number of samples
sigma_train=0.5;  # Stochastic noise
# equally spaced samples between 0 and 1
x=np.linspace(0,1,N_train)
# array containing the gaussian random noise of amplitude  sigma_train
s = sigma_train*np.random.randn(N_train)

################################################
# Define and plot the underlying law
# linear polynomial
y = 2*x
# Here we also add the gaussian noise
ytr = y + s

# Tenth order polynomial
#y=2*x-10*x**5+15*x**10+s

p1=plt.plot(x, y, "D", ms=8, alpha=0.5, label='Underlying Law')
p1=plt.plot(x, ytr, "o", ms=8, alpha=0.5, label='Training Data')

filename_p1="Lecture1-fitting-plt1.pdf"
plt.legend(fontsize=15)
plt.savefig(filename_p1)
exit()

################################################
# Now carry out linear regression
# Linear Regression : create linear regression object
clf = linear_model.LinearRegression()

# Train the model using the training set
# Note: sklearn requires a design matrix of shape (N_train, N_features). Thus we reshape x to (N_train, 1):
clf.fit(x[:, np.newaxis], y)

# Use fitted linear model to predict the y value:
xplot=np.linspace(0.02,0.98,200) # grid of points, some are in the training set, some are not
linear_plot=plt.plot(xplot, clf.predict(xplot[:, np.newaxis]), label='Linear')

# Polynomial Regression
poly3 = PolynomialFeatures(degree=3)
# Construct polynomial features
X = poly3.fit_transform(x[:,np.newaxis])
clf3 = linear_model.LinearRegression()
clf3.fit(X,y)


Xplot=poly3.fit_transform(xplot[:,np.newaxis])
poly3_plot=plt.plot(xplot, clf3.predict(Xplot), label='Poly 3')

# Fifth order polynomial in case you want to try it out
#poly5 = PolynomialFeatures(degree=5)
#X = poly5.fit_transform(x[:,np.newaxis])
#clf5 = linear_model.LinearRegression()
#clf5.fit(X,y)

#Xplot=poly5.fit_transform(xplot[:,np.newaxis])
#plt.plot(xplot, clf5.predict(Xplot), 'r--',linewidth=1)

poly10 = PolynomialFeatures(degree=10)
X = poly10.fit_transform(x[:,np.newaxis])
clf10 = linear_model.LinearRegression()
clf10.fit(X,y)

Xplot=poly10.fit_transform(xplot[:,np.newaxis])
poly10_plot=plt.plot(xplot, clf10.predict(Xplot), label='Poly 10')

plt.legend(loc='best')
plt.ylim([-7,7])
plt.xlabel("x")
plt.ylabel("y")
Title="N=%i, $\sigma=%.2f$"%(N_train,sigma_train)
plt.title(Title+" (train)")


# Linear Filename
filename_train="train-linear_N=%i_noise=%.2f.pdf"%(N_train, sigma_train)

# Tenth Order Filename
#filename_train="train-o10_N=%i_noise=%.2f.pdf"%(N_train, sigma_train)

# Saving figure and showing results
plt.savefig(filename_train)
plt.grid()
plt.show()
