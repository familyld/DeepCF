import numpy as np
import tensorflow as tf
from keras import initializers
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dot, Lambda, multiply, Reshape, merge
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import backend as K
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--userlayers', nargs='?', default='[512, 64]',
                        help="Size of each user layer")
    parser.add_argument('--itemlayers', nargs='?', default='[1024, 64]',
                        help="Size of each item layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def get_model(train, num_users, num_items, userlayers=[512, 64], itemlayers=[1024, 64]):
    num_layer = len(userlayers)  # Number of layers in the MLP
    user_matrix = K.constant(getTrainMatrix(train))
    item_matrix = K.constant(getTrainMatrix(train).T)

    # Input variables
    user = Input(shape=(1,), dtype='int32', name='user_input')
    item = Input(shape=(1,), dtype='int32', name='item_input')

    # Multi-hot User representation and Item representation
    user_input = Lambda(lambda x: tf.gather(user_matrix, tf.to_int32(x)))(user)
    item_input = Lambda(lambda x: tf.gather(item_matrix, tf.to_int32(x)))(item)
    user_input = Reshape((num_items, ))(user_input)
    item_input = Reshape((num_users, ))(item_input)
    print(user_input.shape, item_input.shape)
    
    # DMF part
    userlayer = Dense(userlayers[0], activation="linear" , name='user_layer0')
    itemlayer = Dense(itemlayers[0], activation="linear" , name='item_layer0')
    user_latent_vector = userlayer(user_input)
    item_latent_vector = itemlayer(item_input)
    print(user_latent_vector.shape, item_latent_vector.shape)
    for idx in range(1, num_layer):
        userlayer = Dense(userlayers[idx], activation='relu', name='user_layer%d' % idx)
        itemlayer = Dense(itemlayers[idx], activation='relu', name='item_layer%d' % idx)
        user_latent_vector = userlayer(user_latent_vector)
        item_latent_vector = itemlayer(item_latent_vector)
        print(user_latent_vector.shape, item_latent_vector.shape)

    predict_vector = multiply([user_latent_vector, item_latent_vector])
    prediction = Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.lecun_normal(), name='prediction')(predict_vector)

    print(prediction.shape)

    model_ = Model(inputs=[user, item],
                   outputs=prediction)
    
    return model_

def getTrainMatrix(train):
    num_users, num_items = train.shape
    train_matrix = np.zeros([num_users, num_items], dtype=np.int32)
    for (u, i) in train.keys():
        train_matrix[u][i] = 1
    return train_matrix

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    userlayers = eval(args.userlayers)
    itemlayers = eval(args.itemlayers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("DMF arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_DMF_%d.h5' %(args.dataset, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(train, num_users, num_items, userlayers, itemlayers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time()-t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    
    # Train model
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input,  labels = get_train_instances(train, num_negatives)
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best DMF model is saved to %s" % model_out_file)
