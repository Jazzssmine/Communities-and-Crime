#Angelina Poole
#COEN 140
#Homework 3

#from random import *
from numpy.linalg import inv
import numpy as np
import math

#np.set_printoptions(threshold=np.nan)
##################################################################
# w = (xT*x)^-1 * xT*y
def linear_regression_closed_training(xdata, ydata):
    xdata_t = np.transpose(xdata)
    a = np.dot(xdata_t, xdata)
    a_inverse = inv(a)
    b = np.dot(xdata_t, ydata)
    w = np.dot(a_inverse, b)
    return w
##################################################################
#ynew = wT * xnew
def regression_closed_testing(w, xdata):
    w_t = np.transpose(w)
    # print "w_t"
    # print np.shape(w_t)
    # print "xdata"
    # print np.shape(xdata)
    xdata_t = np.transpose(xdata)
    #ynew = np.dot(w_t, xdata)
    # print "ynew shape"
    ynew = np.dot(w_t, xdata_t)
    return np.transpose(ynew)

##################################################################
# w = (xT*x + lambda*I)^-1 xTy
def ridged_regression_closed_training(xdata, ydata, lambdaval):
  xdata_t = np.transpose(xdata)
  a = np.dot(xdata_t, xdata)
  identitymatrix = np.identity(96)
  b = np.dot(lambdaval, identitymatrix)
  paren_sum = np.add(a, b)
  d = inv(paren_sum)
  e = np.dot(d, xdata_t)
  w = np.dot(e, ydata)
  return w
##################################################################
#ynew = xnew * w
def regression_closed_testingnew(w, xdata):
    ynew = np.dot(xdata, w)
    return ynew
##################################################################
def RMSE(ynew, ytrue):
    return np.sqrt(((ynew - ytrue) ** 2).mean())
##################################################################
#Put the data into a list
trd = []
# with open('crime-train.txt','r') as input_file:
#     for line in input_file:
#         line = line.split()
#         train_data.append(line)
with open('crime-train.txt','r') as input_file:
    input_file.next()
    for line in input_file:
        line = line.split()
        trd.append(line)
print "rows including the row of col names"
train_rows = len(trd)
#print train_rows

print "columns"
train_cols = len(trd[0])
#print train_cols

print "original train numpy"
train_numpy = np.asarray(trd)
print train_numpy
train_numpy = train_numpy.astype(float)
print train_numpy
print np.shape(train_numpy)

yvalue_train = train_numpy[:,0]
print "yvalue_train"
yvalue_train = yvalue_train.reshape(1595, 1)
print yvalue_train

#print "train ones"
ones_train = np.ones((1595, 1))
#print ones_train
#print ones_train.reshape(1, 1595)

train_numpy1 = train_numpy
print "train_numpy1"
print train_numpy1

print "train numpy with 1s column added and 1st col removed"
train_numpy = np.append(train_numpy, ones_train, axis = 1)
train_numpy = train_numpy[:,1:97]
print train_numpy

##################################################################

tstd = []
with open('crime-test.txt','r') as input_file:
    input_file.next()
    for line in input_file:
        line = line.split()
        tstd.append(line)

#print "original test numpy"
test_numpy = np.asarray(tstd)
#print test_numpy
test_numpy = test_numpy.astype(float)
# print test_numpy
# print np.shape(test_numpy)

#print "test ones"
ones_test = np.ones((399, 1))
#print ones_test
#print ones_train.reshape(1, 1595)

# print "test numpy with 1s column added and 1st col removed"
test_numpy = np.append(test_numpy, ones_test, axis = 1)

#save the first column into yvalue_test
yvalue_test = test_numpy[:,0]
# print "yvalue_test"
yvalue_test = yvalue_test.reshape(399, 1)
#print yvalue_test

test_numpy = test_numpy[:,1:97]
# print test_numpy

##################################################################
#TRAINING
#LINEAR REGRESSION CLOSED
weight_lr_train = linear_regression_closed_training(train_numpy, yvalue_train)
#weight_lr_test = linear_regression_closed_training(test_numpy, yvalue_train)
# print "weight_lr_train"
# print np.shape(weight_lr_train)
#print weight_lr_train
##################################################################
#TESTING
#LINEAR REGRESSION CLOSED
#Predicted values using
predicted_values = regression_closed_testing(weight_lr_train, test_numpy)
predicted_values2 = regression_closed_testing(weight_lr_train, train_numpy)

print "predicted values2"
#print predicted_values2
print np.shape(predicted_values)
##################################################################
#ERROR RATES for Linear Regression Closed Form

rmse_train = RMSE(predicted_values2, yvalue_train)
rmse_test = RMSE(predicted_values, yvalue_test)
print "##################################################################"
print "Error Rates for Linear Regression Closed Form:"
print "RMSE Train is %s" %(rmse_train)
print "RMSE Test is %s" %(rmse_test)
print "##################################################################"
##################################################################
# #RIDGE REGRESSION CLOSED
# #5 fold cross validation

#divide training data up into 5 sections
train_numpy_section1 = train_numpy[0:319]
train_yvalue_section1 = yvalue_train[0:319]
# print "shape 1"
# print np.shape(train_numpy_section1)

train_numpy_section2 = train_numpy[319:638]
train_yvalue_section2 = yvalue_train[319:638]
# print "shape 2"
# print np.shape(train_numpy_section2)

train_numpy_section3 = train_numpy[638:957]
train_yvalue_section3 = yvalue_train[638:957]
# print "shape 3"
# print np.shape(train_numpy_section3)

train_numpy_section4 = train_numpy[957:1276]
train_yvalue_section4 = yvalue_train[957:1276]
# print "shape 4"
# print np.shape(train_numpy_section4)

train_numpy_section5 = train_numpy[1276:1596]
train_yvalue_section5 = yvalue_train[1276:1596]
# print "shape 5"
# print np.shape(train_numpy_section5)

#####################################
#round 1: train (1, 2, 3, 4) | test 5
round1_train = train_numpy_section1
round1_train = np.append(round1_train, train_numpy_section2, axis = 0)
round1_train = np.append(round1_train, train_numpy_section3, axis = 0)
round1_train = np.append(round1_train, train_numpy_section4, axis = 0)

round1_y = train_yvalue_section1
round1_y = np.append(round1_y, train_yvalue_section2, axis = 0)
round1_y = np.append(round1_y, train_yvalue_section3, axis = 0)
round1_y = np.append(round1_y, train_yvalue_section4, axis = 0)

#####################################
#round 2: train (1, 2, 3, 5) | test 4
round2_train = train_numpy_section1
round2_train = np.append(round2_train, train_numpy_section2, axis = 0)
round2_train = np.append(round2_train,train_numpy_section3, axis = 0)
round2_train = np.append(round2_train,train_numpy_section5, axis = 0)

round2_y = train_yvalue_section1
round2_y = np.append(round2_y, train_yvalue_section2, axis = 0)
round2_y = np.append(round2_y, train_yvalue_section3, axis = 0)
round2_y = np.append(round2_y, train_yvalue_section5, axis = 0)

#####################################
#round 3: train (1, 2, 4, 5) | test 3
round3_train = train_numpy_section1
round3_train = np.append(round3_train, train_numpy_section2, axis = 0)
round3_train = np.append(round3_train, train_numpy_section4, axis = 0)
round3_train = np.append(round3_train, train_numpy_section5, axis = 0)

round3_y = train_yvalue_section1
round3_y = np.append(round3_y, train_yvalue_section2, axis = 0)
round3_y = np.append(round3_y, train_yvalue_section4, axis = 0)
round3_y = np.append(round3_y, train_yvalue_section5, axis = 0)

#####################################
#round 4: train (1, 3, 4, 5) | test 2
round4_train = train_numpy_section1
round4_train = np.append(round4_train, train_numpy_section3, axis = 0)
round4_train = np.append(round4_train, train_numpy_section4, axis = 0)
round4_train = np.append(round4_train, train_numpy_section5, axis = 0)

round4_y = train_yvalue_section1
round4_y = np.append(round4_y, train_yvalue_section3, axis = 0)
round4_y = np.append(round4_y, train_yvalue_section4, axis = 0)
round4_y = np.append(round4_y, train_yvalue_section5, axis = 0)

#####################################
#round 5: train (2, 3, 4, 5) | test 1
round5_train = train_numpy_section2
round5_train = np.append(round5_train, train_numpy_section3, axis = 0)
round5_train = np.append(round5_train, train_numpy_section4, axis = 0)
round5_train = np.append(round5_train, train_numpy_section5, axis = 0)

round5_y = train_yvalue_section2
round5_y = np.append(round5_y, train_yvalue_section3, axis = 0)
round5_y = np.append(round5_y, train_yvalue_section4, axis = 0)
round5_y = np.append(round5_y, train_yvalue_section5, axis = 0)

# ##################################################################
#trying to find optimal lambda RIDGED REGRESSION CLOSED FORM

average_error = 1
temp_error = 1
lambdaval = float(400)

for i in range(0, 10):
  print "Lambda is: %s" %(lambdaval)
  print "Round 1"
  #print round1_train
  w1 = ridged_regression_closed_training(round1_train, round1_y, lambdaval)

  pred_value1 = regression_closed_testingnew(w1, train_numpy_section5)
  print "predicted value round 1"
  #print pred_value1

  e1 = RMSE(pred_value1, train_yvalue_section5)
  print "Error 1 is: %s" %(e1)
  print "Round 2"
  w2 = ridged_regression_closed_training(round2_train, round2_y, lambdaval)
  #print w2
  pred_value2 = regression_closed_testingnew(w2, train_numpy_section4)
  print "predicted value round 2"
  #print pred_value2

  e2 = RMSE(pred_value2, train_yvalue_section4)
  print "Error 2 is: %s" %(e2)
  print "Round 3"
  w3 = ridged_regression_closed_training(round3_train, round3_y, lambdaval)
  #print "w3"
  #print w3
  pred_value3 = regression_closed_testingnew(w3, train_numpy_section3)
  print "predicted value round 3"
  #print pred_value3

  e3 = RMSE(pred_value3, train_yvalue_section3)
  print "Error 3 is: %s" %(e3)
  print "Round 4"
  w4 = ridged_regression_closed_training(round4_train, round4_y, lambdaval)
  print "w4"
  #print w4
  pred_value4 = regression_closed_testingnew(w4, train_numpy_section2)
  #print "predicted value round 4"
  #print pred_value4

  e4 = RMSE(pred_value4, train_yvalue_section2)
  print "Error 4 is: %s" %(e4)
  print "Round 5"
  w5 = ridged_regression_closed_training(round5_train, round5_y, lambdaval)
  #print "w5"
  # print w5
  print "weight round 5"
  pred_value5 = regression_closed_testingnew(w5, train_numpy_section1)
  #print "predicted value round 5"
  #print pred_value5

  e5 = RMSE(pred_value5, train_yvalue_section1)
  print "Error 5 is: %s" %(e5)
  average_error = (e1 + e2 + e3 + e4 + e5)/5
  print "Average error rate for lambda of %s is: %s" %(lambdaval,average_error)

  if average_error < temp_error:
    temp_error = average_error
    optimal_lambda = lambdaval
    print "temp_error"
    print temp_error

  lambdaval = lambdaval/2

print "The optimal lambda for ridged regression closed form: %s" %(optimal_lambda)

##################################################################
#TRAINING WITH OPTIMAL LAMBDA
#RIDGE REGRESSION CLOSED
optimal_lambda = 25
weight_rr_train = ridged_regression_closed_training(train_numpy, yvalue_train, optimal_lambda)

##################################################################
#TESTING
#RIDGE REGRESSION CLOSED
pval_ridge = regression_closed_testing(weight_rr_train, test_numpy)
pval_ridge_2 = regression_closed_testing(weight_rr_train, train_numpy)

##################################################################
#ERROR RATES for Ridge Regression Closed Form
rr_rmse_test = RMSE(pval_ridge, yvalue_test)
rr_rmse_test_2 = RMSE(pval_ridge_2, yvalue_train)

print "##################################################################"
print "Error Rates for Ridge Regression Closed Form Test Data with Optimal Lambda: %s" %(optimal_lambda)
print "RMSE Train is %s" %(rr_rmse_test_2)
print "RMSE Test is %s" %(rr_rmse_test)
print "##################################################################"
##################################################################
def lossFunc(xdata, ydata, wdata):
  left = np.subtract(np.dot(xdata, wdata), ydata)
  left_t = np.transpose(left)
  return np.dot(left_t, left)

##################################################################
# wt+1 = w^t + alpha*xT*(y-xw^t)
def linear_regression_gradient(xdata, ydata, alpha):
  #w_curr = w^t
  xdata_t = np.transpose(xdata)
  #w0 is N(0,1)
  it = 0
  w_curr= np.random.normal( 0, 1, (96, 1))
  middle_part = np.multiply(alpha, xdata_t)
  right_part = np.subtract(ydata, np.dot(xdata, w_curr))
  total_right_part = np.dot(middle_part, right_part)
  #w1 = w0 + alpha*xT*(y-xw0)
  w_next = np.add(w_curr, total_right_part)
  it = 1
  #w1 onwards
  diff = 1
  #.0000001 = tolerance
  while(abs(diff) > .0000001):
    w_curr = w_next
    middle_part = np.multiply(alpha, xdata_t)
    right_part = np.subtract(ydata, np.dot(xdata, w_curr))
    total_right_part = np.dot(middle_part, right_part)
    w_next = np.add(w_curr, total_right_part)
    diff = np.amin(np.subtract(w_next, w_curr),0)
    #print "loss function"
    #print lossFunc(xdata, ydata, w_next)

  return w_next
##################################################################
#TRAINING Linear Regression Gradient Descent
print "LINEAR REGRESSION GRADIENT DESCENT"
weight_lrg_train = linear_regression_gradient(train_numpy, yvalue_train, 0.00001)
print "Linear Regression Gradient"
print weight_lrg_train
##################################################################
#ERROR RATES for Linear Regression Gradient Descent
pval_linear_regression_train = regression_closed_testing(weight_lrg_train, train_numpy)
pval_linear_regression_test = regression_closed_testing(weight_lrg_train, test_numpy)

lrgd_rmse_train = RMSE(pval_linear_regression_train, yvalue_train)
lrgd_rmse_test = RMSE(pval_linear_regression_test, yvalue_test)

print "##################################################################"
print "Error Rates for Linear Regression Gradient Descent"
print "RMSE Train is %s" %(lrgd_rmse_train)
print "RMSE Test is %s" %(lrgd_rmse_test)
print "##################################################################"

##################################################################
# wt+1 = wt + alpha(xT*(y-xwt) - lambda*w^t)
def ridged_regression_gradient(xdata, ydata, alpha, lambdaval):
  #w_curr = w^t
  xdata_t = np.transpose(xdata)
  #w0 is N(0,1)
  it = 0
  w_curr= np.random.uniform(0, 1, (96, 1))
  far_right = np.multiply(lambdaval, w_curr)
  right_part = np.subtract(ydata, np.dot(xdata, w_curr))
  middle_part = np.dot(xdata_t, right_part)
  paren_exp = np.subtract(middle_part, far_right)
  total_right_part = np.multiply(alpha, paren_exp)
  #w1 = w0 + alpha*xT*(y-xw0)
  w_next = np.add(w_curr, total_right_part)
  it = 1
  #w1 onwards
  diff = 1
  #.0000001 = tolerance
  while(abs(diff) > .00001):
    w_curr = w_next
    middle_part = np.multiply(alpha, xdata_t)
    right_part = np.subtract(ydata, np.dot(xdata, w_curr))
    total_right_part = np.dot(middle_part, right_part)
    w_next = np.add(w_curr, total_right_part)
    diff = np.amin(np.subtract(w_next, w_curr),0)
    print "ridge regression loss function"
    print lossFunc(xdata, ydata, w_next)

  return w_next

##################################################################
#trying to find optimal lambda RIDGED REGRESSION GRADIENT DESCENT
average_error2 = 1
temp_error2 = 1
thealpha = 0.00001
lambdaval2 = float(400)
print "GETTING OPTIMAL LAMBDA RIDGE REGRESSION GRADIENT DESCENT"
for i in range(0, 10):
  print "Lambda is: %s" %(lambdaval2)
  print "Round 1"
  #print round1_train
  w1 = ridged_regression_gradient(round1_train, round1_y, thealpha, lambdaval2)

  pred_value1 = regression_closed_testing(w1, train_numpy_section5)
  print "predicted value round 1"
  #print pred_value1

  e1 = RMSE(pred_value1, train_yvalue_section5)
  print "Error 1 is: %s" %(e1)
  print "Round 2"
  w2 = ridged_regression_gradient(round2_train, round2_y, thealpha, lambdaval2)
  #print w2
  pred_value2 = regression_closed_testing(w2, train_numpy_section4)
  print "predicted value round 2"
  #print pred_value2

  e2 = RMSE(pred_value2, train_yvalue_section4)
  print "Error 2 is: %s" %(e2)
  print "Round 3"
  w3 = ridged_regression_gradient(round3_train, round3_y, thealpha, lambdaval2)
  #print "w3"
  #print w3
  pred_value3 = regression_closed_testing(w3, train_numpy_section3)
  print "predicted value round 3"
  #print pred_value3

  e3 = RMSE(pred_value3, train_yvalue_section3)
  print "Error 3 is: %s" %(e3)
  print "Round 4"
  w4 = ridged_regression_gradient(round4_train, round4_y, thealpha, lambdaval2)
  print "w4"
  #print w4
  pred_value4 = regression_closed_testing(w4, train_numpy_section2)
  #print "predicted value round 4"
  #print pred_value4

  e4 = RMSE(pred_value4, train_yvalue_section2)
  print "Error 4 is: %s" %(e4)
  print "Round 5"
  w5 = ridged_regression_gradient(round5_train, round5_y, thealpha, lambdaval2)
  #print "w5"
  # print w5
  print "weight round 5"
  pred_value5 = regression_closed_testing(w5, train_numpy_section1)
  #print "predicted value round 5"
  #print pred_value5

  e5 = RMSE(pred_value5, train_yvalue_section1)
  print "Error 5 is: %s" %(e5)
  average_error2 = (e1 + e2 + e3 + e4 + e5)/5
  print "Average error rate for lambda of %s is: %s" %(lambdaval2,average_error2)

  if average_error2 < temp_error:
    temp_error2 = average_error2
    optimal_lambda = lambdaval2
    print "temp_error"
    print temp_error2

  lambdaval2 = lambdaval2/2

print "The optimal lambda for ridged regression gradient descent: %s" %(optimal_lambda)
##################################################################
#TRAINING Ridged Regression Gradient descent
print "RIDGED REGRESSION GRADIENT DESCENT"
optimal_lambda = 25
weight_rrg_train = ridged_regression_gradient(train_numpy, yvalue_train, 0.00001, optimal_lambda)
print "Ridged Regression Gradient Weight"
print weight_rrg_train
##################################################################
#ERROR RATES for Ridged Regression Gradient Descent
pval_ridged_regression_train = regression_closed_testing(weight_rrg_train, train_numpy)
pval_ridged_regression_test = regression_closed_testing(weight_rrg_train, test_numpy)

rrgd_rmse_train = RMSE(pval_ridged_regression_train, yvalue_train)
rrgd_rmse_test = RMSE(pval_ridged_regression_test, yvalue_test)

print "##################################################################"
print "Error Rates for Ridged Regression Gradient Descent"
print "RMSE Train is %s" %(rrgd_rmse_train)
print "RMSE Test is %s" %(rrgd_rmse_test)
print "##################################################################"
