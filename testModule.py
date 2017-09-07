import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, Embedding, merge, Reshape, Embedding, Flatten, Dropout, Lambda
import keras.engine.topology
from keras.engine.topology import Layer
from keras.optimizers import Adam, SGD
import keras.backend as K
import RBFPooling
import GetEmbeddingData
from keras.utils import plot_model
import codecs
from keras.engine import training
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, Callback
from scipy import linalg, mat, dot
from random import shuffle
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import re
import theano.tensor as T

import theano
from scipy import  dot, linalg

UNK_MARK = '!UNK!'
PADDING_MARK = '!PADDING!'
MENTION_MARK = '!MENTION!'
PRED_UNK_MARK = '!PREDUNK!'
INPUT_SIZE = 30
OUT_SIZE = 1
EMBEDD_DIM = 100
MAX_EPOCH = 10
theano.config.floatX = 'float64'
BATCH_SIZE = 200
DROPING = 0.5


def split_data(training_data):
    X_q,X_p, X_n, Y = training_data
    X = [np.asarray(X_q), np.asarray(X_p), np.asarray(X_n)]
    Y = np.array(Y)

    split_at = int(len(Y) * (1.0 - 0.1))
    train_X, val_X = training._slice_arrays(X, 0, split_at), training._slice_arrays(X, split_at)
    train_Y = Y[0:split_at]
    val_Y =Y[split_at:]
    #train_Y, val_Y = training._slice_arrays(Y, 0, split_at), training._slice_arrays(Y, split_at)
    return train_X, val_X, train_Y, val_Y

def charadelt(str):
    word_list = list()

    chineseChara = re.compile(u'[\u4e00-\u9fa5]+')
    for c in str:
        if chineseChara.search(c):
            word_list.append(c)
    return word_list

def load_data(inf_path):
    inf = codecs.open(inf_path, encoding='utf-8', mode='r')
    outputs = []
    pos_dict = dict()
    neg_dict = dict()

    count = 0
    for l in inf:
        if count > 30000000:
            break
        l = l.rstrip('\r\n')
        vals = l.split('\t')
        if len(vals) < 3:
            print 'invalid line: ', l
            continue
        label = vals[0]
        query = vals[1]
        answer = vals[2]

        if label == '1':
            if not pos_dict.has_key(query):
                pos_dict[query] =[]
            pos_dict[query].append(answer)

        else:
            if not neg_dict.has_key(query):
                neg_dict[query] =[]
            neg_dict[query].append(answer)
        count+=1

    inf.close()
    q_count = 0
    for q in pos_dict.keys():
        if len(q) == 0:
            continue
        pos_cases = pos_dict.get(q, [])
        neg_cases = neg_dict.get(q, [])
        for pc in pos_cases:
            for nc in neg_cases:
                outputs.append([q, pc, nc])
    return outputs

def load_test_data(inf_path):
    inf = codecs.open(inf_path, encoding='utf-8', mode='r')
    outputs = []
    pos_dict = dict()
    neg_dict = dict()
    pos_a = []
    for l in inf:
        l = l.rstrip('\r\n')
        vals = l.split('\t')
        if len(vals) != 3:
            print 'invalid line: ', l
        label = vals[0]
        query = vals[1]
        answer = vals[2]

        if label == '1':
            if not pos_dict.has_key(query):
                pos_dict[query] = []
            pos_dict[query].append(answer)
        else:
            if not neg_dict.has_key(query):
                neg_dict[query] = []
            neg_dict[query].append(answer)
    inf.close()
    for q in pos_dict.keys():
        pos_cases = pos_dict.get(q, [])
        neg_cases = neg_dict.get(q, [])
     #   if len(pos_cases) != 1:
     #       print 'invalid test data: ', q
        all_cases = []
        all_cases.extend(pos_cases)
        all_cases.extend(neg_cases)
        pos_a.extend(pos_cases)
        outputs.append([q, all_cases])
    return outputs, pos_a

def load_Embedding_data(inf_path):
    embedding_index = dict()
    word_dict = dict()
    word_index = 0
    emdWeights = []
    word_dict[PADDING_MARK] = word_index
  #  emdWeights.append([0.0 for i in range(0, EMBEDD_DIM)])
    word_index +=1

    f = codecs.open(inf_path, encoding='utf-8', mode='r' )
    f.readline()
    count = 0
    for l in f:
 #       if count > 1000:
 #           break
        count +=1
        vals =l.rstrip('\r\n').split(' ')
        if(len(vals) <2):
            continue
        d = [float(x) for x in vals[1:]]
        while len(d) < EMBEDD_DIM:
            d.append(0)
        '''
        d = []
        if len(d) !=EMBEDD_DIM:
            d = [0.0 for i in range(0, EMBEDD_DIM)]
        '''
     #   emdWeights.append(d)
        word_dict[vals[0]] = word_index
        word_index += 1

    f.close()
    # UNK
    word_dict[UNK_MARK] = word_index
 #   emdWeights.append([np.random.random() - 0.5 for i in range(0, EMBEDD_DIM)])
    word_index += 1

    # MENTION MARK
    word_dict[MENTION_MARK] = word_index
  #  emdWeights.append([np.random.random() - 0.5 for i in range(0, EMBEDD_DIM)])
    word_index += 1

  #  emdWeights.append([0.0 for i in range(0, EMBEDD_DIM)])
#    emdWeights = [np.asarray(emdWeights)]
    print('Found %s word vectors.' % len(emdWeights))
    print  'Found words:', len(word_dict)
  #  return word_dict, emdWeights
    return word_dict

def loadTrain_data(file_dict, embeddweight):
    q_dict =[]
    pos_dict = []
    neg_dict = []
    word_dict = []

    tuples = []
    count = 0
    for x in file_dict:
        if count > 5:
            break
        count +=1
        query = x[0]
        pos = x[1]
        neg = x[2]
        query_words = query.split(' ')
        pos_words = pos.split(' ')
        neg_words = neg.split(' ')

        query_embedd =[]
        for i in query_words:
            if i not in word_dict :
                word_dict.append(i)
            if embeddweight.has_key(i):
                query_embedd.append(embeddweight[i])
        pos_embedd =[]
        for i in pos_words:
            if i not in word_dict:
                word_dict.append(i)
            if embeddweight.has_key(i):
                pos_embedd.append(embeddweight[i])

        neg_embedd =[]
        for i in neg_words:
            if i not in word_dict:
                word_dict.append(i)
            if embeddweight.has_key(i):
                neg_embedd.append(embeddweight[i])
        query_embedd = np.array(query_embedd, dtype = float)
        pos_embedd = np.array(pos_embedd, dtype = float)
        neg_embedd = np.array(neg_embedd, dtype = float)

        if query_embedd.size == 0 or pos_embedd.size == 0 or neg_embedd.size == 0:
            continue

        tuple = [query_embedd, pos_embedd, neg_embedd]
        tuple.append(0)
        tuples.append(tuple)
        if count % 1000 == 0:
            print count
    shuffle(tuples)
    q,p,n,y = zip(*tuples)
    return q,p,n,y, word_dict

def get_padding_vector(text, dic, max_len, unk):
    unk_index = dic[unk]
    words = text.split(' ')
    #words = charadelt(text)
    words = [w for w in words if len(w) > 0]
    vec = []
    for w in words:
        if len(vec) < max_len:
            vec.append(dic.get(w, unk_index))
         #  if dic.has_key(w):
         #       vec.append(dic[w])
    while len(vec) < max_len:
        vec.append(0)
    return vec

def get_training_data(X_raw, dic):
    tuples = []
    count = 0
    for x_raw in X_raw:
        tuple = [get_padding_vector(x_raw[0], dic, INPUT_SIZE, UNK_MARK), get_padding_vector(x_raw[1], dic, INPUT_SIZE, UNK_MARK), get_padding_vector(x_raw[2], dic, INPUT_SIZE, UNK_MARK)]
        tuple.append(0.0)
        tuples.append(tuple)
        count += 1
        if count > 20000000:
            break
        if count % 1000 == 0:
            print count
    shuffle(tuples)
    X_q, X_p, X_n, Y = zip(*tuples)
    return X_q, X_p, X_n, Y

def get_slice_data(x, val_split):
    x = np.asarray(x)
    split_at = int(len(x) * (1.0 - val_split))
    train_x = x[0:split_at]
    val_x = x[split_at:]
  #  train_x, val_x =  x[0:split_at], val_Y =x[split_at:]
    return train_x, val_x

def dcg_score(y_true, y_pred, k = 10, gains ='exponential'):
    y_pred = np.asarray(y_pred)
    order = np.argsort(y_pred)
    y_true  = np.asarray(y_true)
    y_true = np.take(y_true, order[:k])

    if gains == 'exponential':
        gains = 2**y_true -1
    elif gains == 'linear':
        gains = y_true
    else:
        raise ValueError('Invalid gains function')

    discounts = np.log2(np.arange(len(y_true)) +2)

    return np.sum(gains/discounts)

def ndcg_score(y_true, y_pred, k = 10, gains = 'exponential'):
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_pred, k, gains)

    return actual/best

class similarity_Cosine(Layer):
    def __init__(self,axis =-1,mask = None, **kwargs):
        self.axis = axis
        self.supports_masking = True
        super(similarity_Cosine, self).__init__( **kwargs)

    def build(self, input_shape):
        self.repeat_dim =  (input_shape[0])[1]
    def call(self, x, mask = None, **kwargs ):
        axis = len(x[0]._keras_shape) - 1
        dot = lambda a, b: K.batch_dot(a, b, axes=axis)
        d = K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1]))
        output = dot(x[0], x[1])/K.maximum(d, K.epsilon())

        if mask is not None:
            mask = K.cast(mask, K.floatx())

            print 'repeat dim', self.repeat_dim
            print 'mask', mask.ndim
          #  mask = K.repeat(mask, self.repeat_dim)
           # mask = mask.dimshuffle(0, 1, 'x')
          #  mask = K.permute_dimensions(mask, (0,2,1))
            output *= mask
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        print input_shape
        n_shape = list(input_shape[0])
        n_shape[-1] = n_shape[1]
        print n_shape
        # return input_shape
        return tuple(n_shape)

def multiply(x,y):
    x_prime = T.reshape(x, (-1, n, 1))
    x_transpose = T.transpose(x_prime, perm=[0, 2, 1])
    return T.batch_matmul(x_transpose, x_prime)

def cosineSimilarity(inputs, axis = -1):
    l1 = inputs[0]
    l2 = inputs[1]
    axis = len(inputs[0]._keras_shape) - 1
    print axis , 'axis'
   # axis = 1

    c = K.sum(l1 * l1, axis=axis, keepdims=True)
    d = K.sum(l2 * l2, axis=axis, keepdims=True)
    e = K.batch_dot(c, d, axis)
    denominator = K.maximum(K.sqrt(e), 1e-7)

    output = K.batch_dot(l1, l2, axis) / denominator
  #  output = K.expand_dims(output, 1)
    return output

def minus(inputs):
    l1 = inputs[0]
    l2 = inputs[1]

    return l1 - l2

def cosine(inputs):
    l1 = inputs[0]
    l2 = inputs[1]
    denominator = K.sqrt(K.batch_dot(l1, l1, 2) *
                         K.batch_dot(l2, l2, 2))
    denominator = K.maximum(denominator, K.epsilon())
    output = K.batch_dot(l1, l2, 2) / denominator
   # output = K.expand_dims(output, 1)
    output = output/K.sum(output, axis= 2, keepdims= True)
    return output
class MyPairwiseLayer(Layer):
    def __init__(self, output_shape, margin=0.5, **kwargs):
        self._output_shape = output_shape
        self._margin = margin
        super(MyPairwiseLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        pos, neg = x
        return K.maximum(self._margin + neg - pos, 0.)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],) + tuple(self._output_shape)

class CustomEmbedding(Embedding):
    def __init__(self, weights_storeable=True, **kwargs):
        self.weights_storeable = weights_storeable
        super(CustomEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CustomEmbedding, self).build(input_shape)
        if not self.weights_storeable:
            self.trainable_weights = []
            self.non_trainable_weights = []

def testModel(WORD_LENGTH, emdWeights = None):
    q_inputs = Input(shape=(INPUT_SIZE,), dtype='int64')
    p_inputs = Input(shape=(INPUT_SIZE,), dtype='int64')
    n_inputs = Input(shape=(INPUT_SIZE,), dtype='int64')

    if emdWeights is None:
        emdQWeights = [np.random.uniform(low=-0.2, high=0.2, size=(WORD_LENGTH, EMBEDD_DIM))]
        emdPWeights = [np.random.uniform(low=-0.2, high=0.2, size=(WORD_LENGTH, EMBEDD_DIM))]
        emdNWeights = [np.random.uniform(low=-0.2, high=0.2, size=(WORD_LENGTH, EMBEDD_DIM))]
    embedding_layer = Embedding(input_dim =WORD_LENGTH, output_dim=EMBEDD_DIM, trainable=True, mask_zero=True, weights=emdQWeights, input_length=None, name='EmbeddQ')

    q_embeddinglayer =  Dropout(p = DROPING)(embedding_layer(q_inputs))
    p_embeddinglayer = Dropout(p = DROPING) (embedding_layer(p_inputs))
    n_embeddinglayer =  Dropout(p = DROPING)(embedding_layer(n_inputs))

   # lstm = LSTM( input_shape=(INPUT_SIZE, EMBEDD_DIM), activation= 'tanh')
   # q_lstm = lstm(q_embeddinglayer)
   # p_lstm = lstm(p_embeddinglayer)
  #  n_lstm = lstm(n_embeddinglayer)

   # translation layer
   # cosinelayer = similarity_Cosine()
   # qp_cosin_sim = cosinelayer([q_embeddinglayer, p_embeddinglayer])
   # qn_cosin_sim = cosinelayer([q_embeddinglayer, n_embeddinglayer])

    qp_cosin_sim = merge([q_embeddinglayer, p_embeddinglayer], mode=cosineSimilarity,  name='cos', output_shape=( INPUT_SIZE, INPUT_SIZE) )
    qn_cosin_sim = merge([q_embeddinglayer, n_embeddinglayer], mode=cosineSimilarity,   name = 'cos1', output_shape=( INPUT_SIZE, INPUT_SIZE) )


  #  reshape = Reshape(target_shape=(INPUT_SIZE, INPUT_SIZE) , name='reshape')
   # qp_reshap = reshape(qp_cosin_sim)
   # qn_reshap = reshape(qn_cosin_sim)



    # kernel layer
    kernel_layer = RBFPooling.RBFLayer(name = 'rbf')
    qp_kernel_layer = kernel_layer(qp_cosin_sim)
    qn_kernel_layer = kernel_layer(qn_cosin_sim)


    '''
    denselayer0 = Dense(11, activation='tanh', name = 'dense0' )
    qp_denselayer0 = denselayer0((qp_kernel_layer))
    qn_denselayer0 = denselayer0((qn_kernel_layer))
    '''


    denselayer = Dense(1, activation='tanh', name='dense1')
    qp_denselayer = Dropout(p = DROPING)( denselayer((qp_kernel_layer)))
    qn_denselayer = Dropout(p = DROPING)(denselayer((qn_kernel_layer)))

   # minuslayer = merge([qp_kernel_layer, qn_kernel_layer], mode='concat',  name='minus', output_shape=( 1,) )


    pairwiselayer = MyPairwiseLayer((1,), margin=0.5)
    final_output = pairwiselayer([qp_denselayer, qn_denselayer])


    model = Model(inputs=[q_inputs, p_inputs, n_inputs], outputs=final_output)

    print model.summary()
    return model

def my_accuracy(y_true, y_pred):
    acc = K.mean(K.equal(y_pred, 0))
    # K.mean()
    # cases = zip(*[y_true, y_pred])
    # if len(cases) == 0:
    #     return 1.0
    # acc = len([c for c in cases if c[0] == c[1]]) / len(cases)
    return acc
def my_accuracy1(y_true, y_pred):
    acc = K.mean(K.equal(y_pred, y_true))
    return acc
def my_accuracy2(y_true, y_pred):
    acc = K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    return acc

def get_pred_vector(pred, pred_dic):
    return [pred_dic.get(pred, pred_dic.get(PRED_UNK_MARK))]

if __name__ == '__main__':
    # data
    '''
    a = np.random.random((10, 1))
    print a
    b = np.random.random((10, 1))
    c = np.random.random((10, 1))

    train_x = [a, b, c]
    y = model.predict(train_x)
    print y, y.shape
    '''



    input_train_file = 'D:/ruizha/sentence/doubanAll.txt'
    input_test_file =  'D:/cache_data/data.test_GCF.reform.split.txt'
    output_file='D:/ruizha/sentence/result.txt'
    embedding_path = 'D:/cache_data/wordembedding_100_skip_win5_min1_epoch3_trunk1.txt'
    output_folder_path ='D:/ruizha/sentence/'

    raw_data = load_data(input_train_file)
    word_dict = load_Embedding_data(embedding_path)
    q, p, n, y = get_training_data(raw_data, word_dict)

    X = [np.asarray(q), np.asarray(p) ,np.asarray(n)]

    WORD_LENGTH = len(word_dict) +1
    Embedding_dim = 100

    model = testModel(WORD_LENGTH)
    adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-7)
    model.compile(loss= 'MSE', optimizer=adam, metrics=['accuracy', my_accuracy, my_accuracy1, my_accuracy2, ndcg_score])

    inp = model.input
    outputs = [layer.get_output_at(0) for layer in model.layers]
    functors = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]

    split_at = 0.1
    train_p, val_p = get_slice_data(p, split_at)
    train_q, val_q = get_slice_data(q, split_at)
    train_n, val_n = get_slice_data(n, split_at)
    train_y, val_y = get_slice_data(y, split_at)

    '''
    score = model.predict([train_q, train_p, train_n])
    of = codecs.open(r'D:/ruizha/sentence/densedataCos.txt', encoding='utf-8', mode='w')

    ll = 0
    for fun in score:
        of.write(str(fun) + r'\n')

    of.close()
    '''
    '''
    of = codecs.open(r'D:/ruizha/sentence/DATA1.txt', encoding='utf-8', mode='w')

    for i in range(len(train_q)):
        of.write(str(train_q[i]) + u'\t' +str(train_p[i])+ u'\t' +str(train_n[i])+ u'\n')
    of.close()
    '''

    '''

    print train_q.shape, train_p.shape, train_n.shape, train_y.shape
    plot_model(model, to_file='model1.png', show_shapes=True)

   # score = model.predict([train_q, train_p, train_n])
    of = codecs.open(r'D:/ruizha/sentence/densedataCos.txt', encoding='utf-8', mode='w')

    ll = 0
    for fun in functors:
        of.write(str(fun) + str(ll))
        ll +=1
        a =  fun([train_q, train_p, train_n,1])
        np.set_printoptions(threshold= None)
        a = np.asarray(a)
        for b in a[0]:
            of.write(str(b) + r'\t')
        of.write(  r'\n')

    of.close()
    '''
  #  np.set_printoptions(threshold=None)
  #  layer_outs = [fun([train_q, train_p, train_n,1]) for fun in functors]
  #  print "layers" ,  layer_outs
 #   print model.get_layer(name='cosine').get_input_shape_at(1), type(
 #       model.get_layer(name='cosine').get_input_at(1))


   # model.fit([train_q, train_p, train_n], train_y, validation_data=([val_q, val_p, val_n], val_y), epochs= MAX_EPOCH, batch_size=100)
    early_stopping = EarlyStopping(patience=2, monitor = 'loss')
    model.fit([train_q, train_p, train_n], train_y, epochs= MAX_EPOCH, batch_size=BATCH_SIZE, shuffle= True, callbacks =[early_stopping] )
    #score = model.evaluate([val_q, val_p, val_n], val_y)

   # score = [0 if x == 0 else 1 for x in score]

   # print metrics.confusion_matrix(val_y, score.round())
   # model.save_weights('result.h5')
    #score = model.evaluate(val_x, val_y)
 #   q_emd_func = K.function(model.inputs, [model.get_layer(name='cos').output])
 #   score_func = K.function(model.get_layer(name='rbf').get_input_at(0), [model.get_layer(name='dense1').get_output_at(0)])
     #                       [model.get_layer(name='dense1').get_output_at(0)])


   # layerput_puts = score_func([train_q, train_p, train_n])[0]

  #  print 'kernel layer outputs: '
   # score = model.predict([train_q, train_p, train_n])
  #  print score

  #  q_cos_func = K.function(model.inputs, [model.get_layer(name='cos').get_output_at(0)])
 #   inputs_tensor = T.stack(model.inputs, axis = 1)
    #model.get_layer(name = 'EmbeddQ').get_input_at(0),model.get_layer(name = 'EmbeddQ').get_input_at(1),model.get_layer(name = 'EmbeddQ').get_input_at(2)
    score_func = K.function(inputs =model.inputs + [K.learning_phase()] , outputs = [model.get_layer(name='dense1').get_output_at(0)])

    print model.get_layer(name='rbf').get_input_shape_at(0), type(
        model.get_layer(name='rbf').get_input_at(0))

    test_data, pos_a = load_test_data(input_test_file)
    of = codecs.open(output_folder_path + 'test.out1.txt', encoding='utf-8', mode='w')
    count = 0
    acc_count = 0
    for td in test_data:
        [q, a_list] = td
        q_vec = get_padding_vector(q, word_dict, INPUT_SIZE, UNK_MARK)
        output = []
        q_emd = None
        for a in a_list:
            a_vec = get_padding_vector(a, word_dict , INPUT_SIZE, UNK_MARK)
          #  pos = get_padding_vector(a_pos, word_dict , INPUT_SIZE, UNK_MARK)
            #a_emd = np.asarray(cos_emd_func([[np.asarray([q_vec]), np.asarray([a_vec]), np.asarray([a_vec])],1])[0])
            score = score_func([np.asarray([q_vec]), np.asarray([a_vec]), np.asarray([a_vec]),1])[0][0]
            output.append([a, score[0]])
        output = sorted(output, key=lambda x: -x[1])

        for o in output:
            of.write(u'\t'.join([q,  o[0] , u':', unicode(o[1]) ]))
            of.write('\t\n')
            if o[0] in pos_a:
                acc_count +=1
            count += 1

        if count % 50 == 0:
            print count
    print acc_count, count
    of.close()



