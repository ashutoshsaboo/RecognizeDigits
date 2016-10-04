import theano as tn
import theano.tensor as T
import numpy as np
import cPickle, gzip

f= gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def shared_dataset(data):
	data_x , data_y = data
	shared_x = tn.shared(np.asarray(data_x, dtype = tn.config.floatX))
	shared_y = tn.shared(np.asarray(data_y, dtype = tn.config.floatX))
	shared_y = T.cast(shared_y,'int32')
	return shared_x, shared_y

test_set_x, test_set_y = shared_dataset(train_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500

data = train_set_x[2*batch_size:3*batch_size]
label = train_set_y[2*batch_size:3*batch_size]

# print data, label

# zero_one_loss = T.sum(T.neq(T.argmax(p_y_given_x), y))
NLL = -1*T.sum(T.log(p_y_given_x)[T.arrange(y.shape[0]),y])

L1 = T.sum(abs(param))
L2 = T.sum(param ** 2)

loss = NLL + lambda_1 * L1 + lambda_2 * L2
# loss = tn.function()
d_loss_wrt_params = T.grad(loss, params)
updates = [(params, params - learning_rate * d_loss_wrt_params)]
MSGD = tn.function([x_batch, y_batch], loss, updates=updates)

for (x_batch, y_batch) in train_batches:
	print "Current Loss is" + MSGD(x_batch, y_batch)
	if stopping_condition_is_met:
		return params

