###############################################################################################
# In this example we walk through the polynomial regression examples
# that we have seen in the lectures
# Original version from arXiv:1803.08823, Phys.Rept. 810 (2019) 1-124
# "A high-bias, low-variance introduction to Machine Learning for physicists"
# P. Mehta, M. Bukov, C.-H. Wang, A. G. R. Day, C. Richardson, C. K. Fisher, D. J. Schwab.
# Adapted by Juan Rojo, j.rojo@vu.nl
#
# -----------------------------------------------------------------------------------------
# Learning goals:
# 1) understand the conceptual differences between training and validation dataset
# 2) demonstrate that fitting does not necessarily imply predicting
# 3) Study empirically the degree of complexity of a given model as a function of the
# properties of the studied data
#
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

# ------------------------------------------------------
# Define the properties of the training dataset
N_train=50  # Number of samples
sigma_train=0.5;  # Stochastic noise
# Define the range spanned by the independent variable
# equally spaced samples between 0 and 1
x=np.linspace(0,1,N_train)
# array containing the gaussian random noise of amplitude  sigma_train
s = sigma_train*np.random.randn(N_train)

################################################
# Define the underlying law (to be learned)

# linear polynomial
# y = 2*x
# Tenth order polynomial
y=2*x-10*x**5+15*x**10

# add the gaussian noise
ytr = y + s

# compare underlying law with training data
plt.plot(x, y, alpha=0.5, label='Underlying Law',linewidth=4.0,markersize=1.0)
plt.plot(x, ytr, "D", ms=8, alpha=0.5, label='Training Data')

filename_p1="Tutorial1-ModelFitting-plt1.pdf"
plt.legend(fontsize=15)
plt.savefig(filename_p1)

################################################
# Now carry out linear regression
# Linear Regression : create linear regression object
clf = linear_model.LinearRegression()

# Train the model using the training set
# Note: sklearn requires a design matrix of shape (N_train, N_features). Thus we reshape x to (N_train, 1)
# since the input variables are defined by a single feature
clf.fit(x[:, np.newaxis], ytr)

# Use fitted linear model to predict the y value:
xplot=np.linspace(0.02,0.98,1000) # grid of points, some are in the training set, some are not
linear_plot=plt.plot(xplot, clf.predict(xplot[:, np.newaxis]), label='Linear')
pred1 = clf.predict(x[:, np.newaxis]) # Same grid as trainig set

################################################
# Polynomial Regression
poly3 = PolynomialFeatures(degree=3) # specify the degree of the polynomial used in the fit
# Construct polynomial features
X = poly3.fit_transform(x[:,np.newaxis])
clf3 = linear_model.LinearRegression()
clf3.fit(X,y)

Xplot=poly3.fit_transform(xplot[:,np.newaxis])
poly3_plot=plt.plot(xplot, clf3.predict(Xplot), label='Poly 3')
pred3 = clf3.predict(poly3.fit_transform(x[:,np.newaxis])) # Same grid as training set
  
# Fifth order polynomial 
poly5 = PolynomialFeatures(degree=5)
X = poly5.fit_transform(x[:,np.newaxis])
clf5 = linear_model.LinearRegression()
clf5.fit(X,y)

Xplot=poly5.fit_transform(xplot[:,np.newaxis])
poly5_plot=plt.plot(xplot, clf5.predict(Xplot), label='Poly 5')
pred5 = clf5.predict(poly5.fit_transform(x[:,np.newaxis])) # Same grid as training set

# Tenth order polynomial 
poly10 = PolynomialFeatures(degree=10)
X = poly10.fit_transform(x[:,np.newaxis])
clf10 = linear_model.LinearRegression()
clf10.fit(X,y)

Xplot=poly10.fit_transform(xplot[:,np.newaxis])
poly10_plot=plt.plot(xplot, clf10.predict(Xplot), label='Poly 10')
pred10 = clf10.predict(poly10.fit_transform(x[:,np.newaxis])) # Same grid as training set

plt.legend(loc='best')
plt.ylim([-3,7])
plt.xlabel("x")
plt.ylabel("y")
Title="N=%i, $\sigma=%.2f$"%(N_train,sigma_train)
plt.title(Title+" (train)")

# Linear Filename
filename_train="Tutorial1-ModelFitting-train-linear_N=%i_noise=%.2f.pdf"%(N_train, sigma_train)

# Tenth Order Filename
#filename_train="train-o10_N=%i_noise=%.2f.pdf"%(N_train, sigma_train)

# Saving figure and showing results
plt.grid()
plt.savefig(filename_train)


# Compute the cost function
i=0
C1tr=0
C1l=0
C3tr=0
C3l=0
C5tr=0
C5l=0
C10tr=0
C10l=0
while (i<N_train):
    C1tr = C1tr + ( pred1[i] - ytr[i])**2 / N_train
    C1l = C1l + ( pred1[i] - y[i])**2 / N_train
    C3tr = C3tr + ( pred3[i] - ytr[i])**2 / N_train
    C3l = C3l + ( pred3[i] - y[i])**2 / N_train
    C5tr = C5tr + ( pred5[i] - ytr[i])**2 / N_train
    C5l = C5l + ( pred5[i] - y[i])**2 / N_train
    C10tr = C10tr + ( pred10[i] - ytr[i])**2 / N_train
    C10l = C10l + ( pred10[i] - y[i])**2 / N_train
    i = i +1

print("C1 (tr), C1 (law) = ", "% 7.5f"% C1tr, "% 7.5f"% C1l) 
print("C3 (tr), C3 (law) = ", "% 7.5f"% C3tr, "% 7.5f"% C3l)
print("C5 (tr), C5 (law) = ", "% 7.5f"% C5tr, "% 7.5f"% C5l)
print("C10 (tr), C10 (law) = ", "% 7.5f"% C10tr, "% 7.5f"% C10l)  


exit()

