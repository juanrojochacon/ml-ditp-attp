###############################################################################################
# In this example  we gain intuition for various gradient descent methods by visualizing and applying these methods to
# some simple two-dimensional surfaces. Methods studied include
# ordinary gradient descent, gradient descent with momentum, NAG, ADAM, and RMSProp.
# Original version from arXiv:1803.08823, Phys.Rept. 810 (2019) 1-124
# "A high-bias, low-variance introduction to Machine Learning for physicists"
# P. Mehta, M. Bukov, C.-H. Wang, A. G. R. Day, C. Richardson, C. K. Fisher, D. J. Schwab.
# Adapted by Juan Rojo, j.rojo@vu.nl
#
# ---------------------------------------------------------------------------------------
# Overview
# ----------
# Here we will visualize what different gradient descent methods are doing using some simple surfaces.
# From the onset, we emphasize that doing gradient descent on the surfaces is different from performing gradient descent on a
# loss function in Machine Learning (ML). The reason is that in ML not only do we want to find good minima, we want
# to find good minima that generalize well to new data. Despite this crucial difference, we can still build
# intuition about gradient descent methods by applying them to simple surfaces.
#
# -----------------------------------------------------------------------------------
#
# Learning goals
# 1) familiarise oneself with the GD method and its variants
# 2) demonstrate how to compare the performance of different minimisers and how to choose the best
# 3) identify the pitfalls that can arise
# 
#####################################################################################################################

print("\n ********* Tutorial Session III: Classifying states in the Ising Model with Random Forests *********** \n")

###############################################################################
# Import the relevant libraries
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt, rcParams
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from IPython.display import HTML
from matplotlib.colors import LogNorm
try:
    # Python 3
    from itertools import zip_longest
except ImportError:
    # Python 2
    from itertools import izip_longest as zip_longest
#############################################################################3

np.random.seed() # shuffle random seed generator

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit

import pickle, os
from urllib.request import urlopen 

# path to data directory (for testing)
#path_to_data=os.path.expanduser('~')+'/Dropbox/MachineLearningReview/Datasets/isingMC/'
url_main = 'https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/';
#path_to_data = '/Users/juanrojo/physics19/ML/IsingData/'

######### Load the Ising Model data
# The data consists of 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25):
data_file_name = "Ising2DFM_reSample_L40_T=All.pkl" 
# The labels are obtained from the following file:
label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"

# Data
data = pickle.load(urlopen(url_main + data_file_name)) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

#LABELS (convention is 1 for ordered states and 0 for disordered states)
labels = pickle.load(urlopen(url_main + label_file_name)) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

print("\n ************** Ising Model data succesfully loaded **************** \n")

###### define ML parameters
from sklearn.model_selection import train_test_split
train_to_test_ratio=0.8 # training samples

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio,test_size=1.0-train_to_test_ratio)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

##### plot a few Ising states

#import ml_style as style
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.rcParams.update(style.style)

from mpl_toolkits.axes_grid1 import make_axes_locatable

# set colourbar map
cmap_args=dict(cmap='plasma_r')

# plot states
fig, axarr = plt.subplots(nrows=1, ncols=3)

axarr[0].imshow(X_ordered[20001].reshape(L,L),**cmap_args)
#axarr[0].set_title('$\\mathrm{ordered\\ phase}$',fontsize=16)
axarr[0].set_title('ordered phase',fontsize=16)
axarr[0].tick_params(labelsize=16)

axarr[1].imshow(X_critical[10001].reshape(L,L),**cmap_args)
#axarr[1].set_title('$\\mathrm{critical\\ region}$',fontsize=16)
axarr[1].set_title('critical region',fontsize=16)
axarr[1].tick_params(labelsize=16)

im=axarr[2].imshow(X_disordered[50001].reshape(L,L),**cmap_args)
#axarr[2].set_title('$\\mathrm{disordered\\ phase}$',fontsize=16)
axarr[2].set_title('disordered phase',fontsize=16)
axarr[2].tick_params(labelsize=16)

fig.subplots_adjust(right=2.0)

plt.show()


# Apply Random Forest

#This is the random forest classifier
from sklearn.ensemble import RandomForestClassifier

#This is the extreme randomized trees
from sklearn.ensemble import ExtraTreesClassifier



#import time to see how perforamance depends on run time

import time

import warnings
#Comment to turn on warnings
warnings.filterwarnings("ignore")

#We will check 

min_estimators = 10
max_estimators = 101
classifer = RandomForestClassifier # BELOW WE WILL CHANGE for the case of extremly randomized forest 

n_estimator_range=np.arange(min_estimators, max_estimators, 10)
leaf_size_list=[2,10000]

m=len(n_estimator_range)
n=len(leaf_size_list)

#Allocate Arrays for various quantities

RFC_OOB_accuracy=np.zeros((n,m))
RFC_train_accuracy=np.zeros((n,m))
RFC_test_accuracy=np.zeros((n,m))
RFC_critical_accuracy=np.zeros((n,m))
run_time=np.zeros((n,m))

print_flag=True

for i, leaf_size in enumerate(leaf_size_list):
    # Define Random Forest Classifier
    myRF_clf = classifer(
        n_estimators=min_estimators,
        max_depth=None, 
        min_samples_split=leaf_size, # minimum number of sample per leaf
        oob_score=True,
        random_state=0,
        warm_start=True # this ensures that you add estimators without retraining everything
    )
    for j, n_estimator in enumerate(n_estimator_range):
        
        print('n_estimators: %i, leaf_size: %i'%(n_estimator,leaf_size))
        
        start_time = time.time()
        myRF_clf.set_params(n_estimators=n_estimator)
        myRF_clf.fit(X_train, Y_train)
        run_time[i,j] = time.time() - start_time

    # check accuracy
        RFC_train_accuracy[i,j]=myRF_clf.score(X_train,Y_train)
        RFC_OOB_accuracy[i,j]=myRF_clf.oob_score_
        RFC_test_accuracy[i,j]=myRF_clf.score(X_test,Y_test)
        RFC_critical_accuracy[i,j]=myRF_clf.score(X_critical,Y_critical)
        if print_flag:
            result = (run_time[i,j], RFC_train_accuracy[i,j], RFC_OOB_accuracy[i,j], RFC_test_accuracy[i,j], RFC_critical_accuracy[i,j])
            print('{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}'.format("time (s)","train score", "OOB estimate","test score", "critical score"))
            print('{0:<15.4f}{1:<15.4f}{2:<15.4f}{3:<15.4f}{4:<15.4f}'.format(*result))

plt.plot(n_estimator_range,RFC_train_accuracy[1],'--b^',label='Train (coarse)')
plt.plot(n_estimator_range,RFC_test_accuracy[1],'--r^',label='Test (coarse)')
plt.plot(n_estimator_range,RFC_critical_accuracy[1],'--g^',label='Critical (coarse)')

plt.plot(n_estimator_range,RFC_train_accuracy[0],'o-b',label='Train (fine)')
plt.plot(n_estimator_range,RFC_test_accuracy[0],'o-r',label='Test (fine)')
plt.plot(n_estimator_range,RFC_critical_accuracy[0],'o-g',label='Critical (fine)')

#plt.semilogx(lmbdas,train_accuracy_SGD,'*--b',label='SGD train')

plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Accuracy')
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("Ising_RF.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

plt.plot(n_estimator_range, run_time[1], '--k^',label='Coarse')
plt.plot(n_estimator_range, run_time[0], 'o-k',label='Fine')
plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Run time (s)')


plt.legend(loc=2)
#plt.savefig("Ising_RF_Runtime.pdf")

plt.show()


#This is the extreme randomized trees
from sklearn.ensemble import ExtraTreesClassifier

#import time to see how perforamance depends on run time

import time

import warnings
#Comment to turn on warnings
warnings.filterwarnings("ignore")

#We will check 


min_estimators = 10
max_estimators = 101
classifer = ExtraTreesClassifier # only changing this

n_estimator_range=np.arange(min_estimators, max_estimators, 10)
leaf_size_list=[2,10000]

m=len(n_estimator_range)
n=len(leaf_size_list)

#Allocate Arrays for various quantities

ETC_OOB_accuracy=np.zeros((n,m))
ETC_train_accuracy=np.zeros((n,m))
ETC_test_accuracy=np.zeros((n,m))
ETC_critical_accuracy=np.zeros((n,m))
run_time=np.zeros((n,m))

print_flag=True

for i, leaf_size in enumerate(leaf_size_list):
    # Define Random Forest Classifier
    myRF_clf = classifer(
        n_estimators=min_estimators,
        max_depth=None, 
        min_samples_split=leaf_size, # minimum number of sample per leaf
        oob_score=True,
        bootstrap=True,
        random_state=0,
        warm_start=True # this ensures that you add estimators without retraining everything
    )
    for j, n_estimator in enumerate(n_estimator_range):
        
        print('n_estimators: %i, leaf_size: %i'%(n_estimator,leaf_size))
        
        start_time = time.time()
        myRF_clf.set_params(n_estimators=n_estimator)
        myRF_clf.fit(X_train, Y_train)
        run_time[i,j] = time.time() - start_time

    # check accuracy
        ETC_train_accuracy[i,j]=myRF_clf.score(X_train,Y_train)
        ETC_OOB_accuracy[i,j]=myRF_clf.oob_score_
        ETC_test_accuracy[i,j]=myRF_clf.score(X_test,Y_test)
        ETC_critical_accuracy[i,j]=myRF_clf.score(X_critical,Y_critical)
        if print_flag:
            result = (run_time[i,j], ETC_train_accuracy[i,j], ETC_OOB_accuracy[i,j], ETC_test_accuracy[i,j], ETC_critical_accuracy[i,j])
            print('{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}'.format("time (s)","train score", "OOB estimate","test score", "critical score"))
            print('{0:<15.4f}{1:<15.4f}{2:<15.4f}{3:<15.4f}{4:<15.4f}'.format(*result))


plt.figure()
plt.plot(n_estimator_range,ETC_train_accuracy[1],'--b^',label='Train (coarse)')
plt.plot(n_estimator_range,ETC_test_accuracy[1],'--r^',label='Test (coarse)')
plt.plot(n_estimator_range,ETC_critical_accuracy[1],'--g^',label='Critical (coarse)')

plt.plot(n_estimator_range,ETC_train_accuracy[0],'o-b',label='Train (fine)')
plt.plot(n_estimator_range,ETC_test_accuracy[0],'o-r',label='Test (fine)')
plt.plot(n_estimator_range,ETC_critical_accuracy[0],'o-g',label='Critical (fine)')

#plt.semilogx(lmbdas,train_accuracy_SGD,'*--b',label='SGD train')

plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Accuracy')
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("Ising_RF.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

plt.plot(n_estimator_range, run_time[1], '--k^',label='Coarse')
plt.plot(n_estimator_range, run_time[0], 'o-k',label='Fine')
plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Run time (s)')


plt.legend(loc=2)
#plt.savefig("Ising_ETC_Runtime.pdf")

plt.show()

exit()
