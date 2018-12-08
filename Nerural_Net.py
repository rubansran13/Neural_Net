import pandas as pd
import numpy as np
MFCC=pd.read_csv("/PATH/Frogs_MFCCs.csv")
columns=list(MFCC.columns)
columns.pop(0)
del columns[21:25]
data=MFCC[columns]
Species_Name=MFCC["Species"]
labels=np.asarray(pd.get_dummies(Species_Name))
labels_class=np.argmax(labels,axis=1)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.2)

from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
SS.fit(X_train)
X_train=SS.transform(X_train)
X_test=SS.transform(X_test)

n_features=len(columns)
n_classes=len(np.unique(labels_class))
Learning_rate=1e-3
epochs=100
first_layer_neurons=25
second_layer_neurons=15
third_layer_neurons=10
batchsize=64


import tensorflow as tf
X = tf.placeholder(tf.float32, [None, n_features], name='features')
Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))

#<----------------First layer with 25 neurons------------->
first_layer_weights=new_weights([n_features,first_layer_neurons])
first_layer_bias=new_biases(first_layer_neurons)
Y1 = tf.nn.tanh((tf.matmul( X,first_layer_weights)+first_layer_bias), name='activationLayer1')

#<----------------second layer with 15 neurons------------>
second_layer_weights=new_weights([first_layer_neurons,second_layer_neurons])
second_layer_bias=new_biases(second_layer_neurons)
Y2 = tf.nn.tanh((tf.matmul(Y1,second_layer_weights)+second_layer_bias),name='activationLayer2')

#<----------------third layer with 10 neurons------------->
third_layer_weights=new_weights([second_layer_neurons,third_layer_neurons])
third_layer_bias=new_biases(third_layer_neurons)
Y3 = tf.nn.tanh((tf.matmul(Y2,third_layer_weights)+third_layer_bias),name='activationLayer2')

#<----------------Softmax layer------------->
fourth_layer_weights = new_weights([ third_layer_neurons,n_classes])
fourth_layer_bias = new_biases(n_classes)
Y_out = tf.nn.softmax((tf.matmul(Y3,fourth_layer_weights) + fourth_layer_bias), name='activationOutputLayer')

#<------------------Cost Function------------------>
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_out),reduction_indices=[1]))


optimizer = tf.train.GradientDescentOptimizer(Learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y_out, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")



sess=tf.Session()
sess.run(tf.global_variables_initializer())

train_cost=[]
test_cost=[]
train_accuracy=[]
test_accuracy=[]

#<------------------Traing model through batch gradient descent------------------>
for e in range(epochs):
        for i in range(len(X_train)):
            start=i
            end=i+batchsize
            x_batch=X_train[start:end]
            y_batch=Y_train[start:end]
            
            # feeding training data/examples
            sess.run(optimizer, feed_dict={X:x_batch , Y:y_batch})
            i+=batchsize
        # feeding testing data to determine model accuracy
        y_pred = sess.run(tf.argmax(Y_out, 1), feed_dict={X: X_train})
        y_true = sess.run(tf.argmax(Y_train, 1))
        train_cost.append(sess.run( cross_entropy, feed_dict={X: X_train, Y: Y_train}))
        test_cost.append(sess.run( cross_entropy, feed_dict={X: X_test, Y: Y_test}))
        
        train_accuracy.append(sess.run( accuracy, feed_dict={X: X_train, Y: Y_train}))
        test_accuracy.append(sess.run( accuracy, feed_dict={X: X_test, Y: Y_test}))
from matplotlib import pyplot as plt
k=[i for i in range(1,len(test_accuracy)+1)]
plt.figure(figsize=(16,12))
plt.subplot(121)
plt.plot(k,test_accuracy,label="Test Accuracy");plt.plot(k,train_accuracy,label="Train Accuracy")
plt.xlabel("Number of iteratios",fontsize=15)
plt.ylabel("Accuracy score",fontsize=15)
plt.legend(loc="upper left")
plt.yticks(np.arange(0.6, 1.05, step=0.05))
plt.title("Accuracy score over iterations",fontsize=20)
plt.subplot(122)
plt.plot(k,test_cost,label="Test Cost");plt.plot(k,train_cost,label="Train Cost")
plt.xlabel("Number of iteratios",fontsize=15)
plt.ylabel("Cost",fontsize=15)
plt.legend(loc="upper right")
plt.title("Cost over iterrations",fontsize=20)
plt.yticks(np.arange(0, 1.2, step=0.1))
plt.savefig("Performance")
