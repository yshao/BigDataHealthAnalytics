import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.contrib import rnn
from tensorflow.core.framework import summary_pb2
import os
import shutil
import sys


def define_network():
    num_static_features = len(static_features_names)
    with tf.name_scope('input'):
        static_features = tf.placeholder(tf.float32, [None, num_static_features], name='static_features')

        lab_values = tf.placeholder(tf.float32, [None, maxSeqLength, len(lab_names)], name='lab_values')

        labels = tf.placeholder(tf.float32, [None], name='labels')
        labels = tf.expand_dims(labels, 1)


    with tf.name_scope('lstm_and_multip'):
        cell = rnn.BasicLSTMCell(rnn_num_hidden, forget_bias=1.0)

        val, _ = tf.nn.dynamic_rnn(cell, lab_values, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        rnn_outputs = tf.gather(val, int(val.get_shape()[0]) - 1)


        initial_value = tf.truncated_normal([rnn_num_hidden, rnn_output_size], stddev=0.001)
        rnn_out_weights = tf.Variable(initial_value, name='rnn_out_weights')
        rnn_out_biases = tf.Variable(tf.zeros([rnn_output_size]), name='rnn_out_biases')

        rnn_fc = tf.matmul(rnn_outputs, rnn_out_weights) + rnn_out_biases
        rnn_fc = tf.nn.relu(rnn_fc)


    with tf.name_scope('static_features_fc'):
        W = tf.get_variable(
            name='W_fc',
            shape=[num_static_features, fc_output_vector_size],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

        b = tf.get_variable(
            name='b_fc',
            shape=[fc_output_vector_size],
            initializer=tf.constant_initializer())

        fc = tf.nn.bias_add(tf.matmul(static_features, W), b)
        fc = tf.nn.relu(fc)


    # concatenate the vectors
    concat_layer = tf.concat([fc, rnn_fc], 1)

    W_2 = tf.get_variable(
        name='W_2',
        shape=[fc_output_vector_size + rnn_output_size, w2_size],
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

    b_2 = tf.get_variable(
        name='b_2',
        shape=[w2_size],
        initializer=tf.constant_initializer())

    fc_2 = tf.nn.bias_add(tf.matmul(concat_layer, W_2), b_2)
    fc_2 = tf.nn.relu(fc_2)





    W_final = tf.get_variable(
        name='W_final',
        shape=[w2_size, 1],
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

    b_final = tf.get_variable(
        name='b_final',
        shape=[1],
        initializer=tf.constant_initializer())

    logits = tf.nn.bias_add(tf.matmul(fc_2, W_final), b_final)
    predictions = tf.sigmoid(logits)

    correct_prediction = tf.equal(tf.round(predictions), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    error = tf.reduce_mean(tf.abs(predictions - labels))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_mean)

    aucroc = tf.contrib.metrics.streaming_auc(predictions, labels)

    return (predictions, optimizer, error, cross_entropy_mean, accuracy, aucroc)


def build_dict(data):
    dictionary = {}

    for subj_id in data.SUBJECT_ID.unique():
        rows = data[data.SUBJECT_ID == subj_id]
        dictionary[subj_id] = {}
        
        dictionary[subj_id]['static_features'] = np.array(rows.iloc[0][static_features_names]).astype(float)

        for lab_name in lab_names:
            values = np.array(rows[lab_name].tail(maxSeqLength))
            initial_vals = np.empty(maxSeqLength - values.shape[0])
            initial_vals.fill(values[0])
            values = np.concatenate((initial_vals, values))
            dictionary[subj_id][lab_name] = values

    return dictionary


def load_data(dataset_to_use):
    # if use_validation_data:
    #     data = pd.read_csv('../data_processed/validation_lab_sequence.csv')
    # else:
    #     data = pd.read_csv('../data_processed/train_lab_sequence.csv')
    if dataset_to_use == 'train':
        data = pd.read_csv('../data_processed/train_final_seq.csv')
    elif dataset_to_use == 'validation':
        data = pd.read_csv('../data_processed/validation_final_seq.csv')
    elif dataset_to_use == 'test':
        data = pd.read_csv('../data_processed/test_final_seq.csv')
    else:
        data = pd.read_csv('../data_processed/zzz.csv')
        
    data_readmission = data[data.readmission == 1]
    data_no = data[data.readmission == 0]

    tf.logging.info('%s  loading readmission data...' % (datetime.now()))
    readmission_dict = build_dict(data_readmission)
    tf.logging.info('%s  loading no readmission data...' % (datetime.now()))
    nonre_dict = build_dict(data_no)

    return readmission_dict, nonre_dict


def get_batch(batchSize, readmission_dict, nonre_dict, readmission_keys, nonre_keys, get_non=True):
    batch_static_features = []
    batch_labs = []
    batch_labels = []
    num_to_use = batchSize
    if not batchSize:
        if get_non:
            num_to_use = len(nonre_keys)
        else:
            num_to_use = len(readmission_keys)

    for j in xrange(num_to_use):
        if (j % 2 == 0 and batchSize) or (not batchSize and get_non):
            if not batchSize:
                subj_id = nonre_keys[j]
            else:
                subj_id = np.random.choice(nonre_keys)
            batch_static_features.append(nonre_dict[subj_id]['static_features'])
            lab_vals = []
            for lab_name in lab_names:
                lab_vals.append(nonre_dict[subj_id][lab_name])
            batch_labels.append(0)
        else:
            if not batchSize:
                subj_id = readmission_keys[j]
            else:
                subj_id = np.random.choice(readmission_keys)
            batch_static_features.append(readmission_dict[subj_id]['static_features'])
            lab_vals = []
            for lab_name in lab_names:
                lab_vals.append(readmission_dict[subj_id][lab_name])
            batch_labels.append(1)
        lab_vals = np.stack(lab_vals)
        lab_vals = np.transpose(lab_vals)
        batch_labs.append(lab_vals)

    batch_static_features = np.array(batch_static_features)
    batch_labs = np.array(batch_labs)
    batch_labels = np.array(batch_labels)
    return (batch_static_features, batch_labs, batch_labels)


def get_total_batch(batchSize, readmission_dict, nonre_dict, readmission_keys, nonre_keys, get_non=True):
    batch_static_features = []
    batch_labs = []
    batch_labels = []
    num_to_use = batchSize
    if not batchSize:
        if get_non:
            num_to_use = len(nonre_keys)
        else:
            num_to_use = len(readmission_keys)
    for typed in ['nonre', 'readmission']:
        keys_to_use = nonre_keys
        dict_to_use = nonre_dict
        label_to_use = 0
        if typed == 'readmission':
            keys_to_use = readmission_keys
            dict_to_use = readmission_dict
            label_to_use = 1

        num_to_use = len(keys_to_use)
        for j in xrange(num_to_use):
            subj_id = keys_to_use[j]
            batch_static_features.append(dict_to_use[subj_id]['static_features'])
            lab_vals = []
            for lab_name in lab_names:
                lab_vals.append(dict_to_use[subj_id][lab_name])
            batch_labels.append(label_to_use)

            lab_vals = np.stack(lab_vals)
            lab_vals = np.transpose(lab_vals)
            batch_labs.append(lab_vals)

    batch_static_features = np.array(batch_static_features)
    batch_labs = np.array(batch_labs)
    batch_labels = np.array(batch_labels)
    return (batch_static_features, batch_labs, batch_labels)


def final_accuracy(dataset_name, sess, merged, accuracy, readmission_dict, nonre_dict, readmission_keys, nonre_keys, train_readmission_writer, train_nonre_writer, train_avg_writer, batch_number, aucroc, is_final):

    batch_static_features, batch_labs, batch_labels = get_batch(None, readmission_dict, nonre_dict, readmission_keys, nonre_keys, True)
    feed_dict = {'input/static_features:0': batch_static_features, 
                 'input/lab_values:0': batch_labs, 
                 'input/labels:0': batch_labels}
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
    # acc = acc[0]
    total_acc = acc
    if train_nonre_writer:
        train_nonre_writer.add_summary(summary, batch_number)
    tf.logging.info('%s  batch_number:%d  %s Accuracy (No Readmission):%f  Total patients:%d' % (datetime.now(), batch_number, dataset_name, acc, batch_labels.shape[0]))

    batch_static_features, batch_labs, batch_labels = get_batch(None, readmission_dict, nonre_dict, readmission_keys, nonre_keys, False)
    feed_dict = {'input/static_features:0': batch_static_features, 
                 'input/lab_values:0': batch_labs, 
                 'input/labels:0': batch_labels}
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
    # acc = acc[0]
    total_acc += acc
    if train_readmission_writer:
        train_readmission_writer.add_summary(summary, batch_number)
    tf.logging.info('%s  batch_number:%d  %s Accuracy (Readmission):%f  Total patients:%d' % (datetime.now(), batch_number, dataset_name, acc, batch_labels.shape[0]))
    tf.logging.info('%s  batch_number:%d  Average %s Accuracy:%f' % (datetime.now(), batch_number, dataset_name, total_acc / 2))

    value = summary_pb2.Summary.Value(tag="accuracy", simple_value=total_acc / 2)
    summary = summary_pb2.Summary(value=[value])
    if train_avg_writer:
        train_avg_writer.add_summary(summary, batch_number)

    if is_final:
        batch_static_features, batch_labs, batch_labels = get_total_batch(None, readmission_dict, nonre_dict, readmission_keys, nonre_keys, False)
        feed_dict = {'input/static_features:0': batch_static_features, 
                     'input/lab_values:0': batch_labs, 
                     'input/labels:0': batch_labels}
        rocval = sess.run([aucroc], feed_dict=feed_dict)
        rocval = rocval[0]
        tf.logging.info('%s  batch_number:%d  %s AUC ROC:%f' % (datetime.now(), batch_number, dataset_name, rocval[1]))


def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    (readmission_dict, nonre_dict) = load_data('train')
    readmission_keys = readmission_dict.keys()
    nonre_keys = nonre_dict.keys()

    tf.logging.info('%s  defining network...' % (datetime.now()))
    (predictions, optimizer, error, cross_entropy_mean, accuracy, aucroc) = define_network()

    tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    sess = tf.Session()
    train_readmission_writer = tf.summary.FileWriter('train/readmission', sess.graph)
    train_nonre_writer = tf.summary.FileWriter('train/nonre')
    train_avg_writer = tf.summary.FileWriter('train/avg')
    validation_readmission_writer = tf.summary.FileWriter('validation/readmission')
    validation_nonre_writer = tf.summary.FileWriter('validation/nonre')
    validation_avg_writer = tf.summary.FileWriter('validation/avg')
    sess.run(tf.global_variables_initializer())

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())



    # final_accuracy('zzz', sess, merged, accuracy, readmission_dict, nonre_dict, readmission_keys, nonre_keys, None, None, None, 0, aucroc, aucpr, True)



    (readmission_dict_valid, nonre_dict_valid) = load_data('validation')
    readmission_keys_valid = readmission_dict_valid.keys()
    nonre_keys_valid = nonre_dict_valid.keys()

    for i in xrange(num_batches):
        batch_static_features, batch_labs, batch_labels = get_batch(16, readmission_dict, nonre_dict, readmission_keys, nonre_keys)

        feed_dict = {'input/static_features:0': batch_static_features, 
                     'input/lab_values:0': batch_labs, 
                     'input/labels:0': batch_labels}

        if i % 100 == 0:
            final_accuracy('train', sess, merged, accuracy, readmission_dict, nonre_dict, readmission_keys, nonre_keys, train_readmission_writer, train_nonre_writer, train_avg_writer, i, aucroc, False)
            # acc = sess.run([accuracy], feed_dict=feed_dict)
            # acc = acc[0]
            # tf.logging.info('%s  i:%d  Accuracy:%f' % (datetime.now(), i, acc))

            final_accuracy('validation', sess, merged, accuracy, readmission_dict_valid, nonre_dict_valid, readmission_keys_valid, nonre_keys_valid, validation_readmission_writer, validation_nonre_writer, validation_avg_writer, i, aucroc, False)

        sess.run([optimizer], feed_dict=feed_dict)

    (readmission_dict_test, nonre_dict_test) = load_data('test')
    readmission_keys_test = readmission_dict_test.keys()
    nonre_keys_test = nonre_dict_test.keys()

    print('\n\n\n')
    final_accuracy('train', sess, merged, accuracy, readmission_dict, nonre_dict, readmission_keys, nonre_keys, None, None, None, i, aucroc, True)
    final_accuracy('validation', sess, merged, accuracy, readmission_dict_valid, nonre_dict_valid, readmission_keys_valid, nonre_keys_valid, None, None, None, i, aucroc, True)
    final_accuracy('test', sess, merged, accuracy, readmission_dict_test, nonre_dict_test, readmission_keys_test, nonre_keys_test, None, None, None, i, aucroc, True)


# static_features_names = ['LOS','GENDER','age','is_urgent','is_emergency']
static_features_names = ['LOS','GENDER','age','is_urgent','is_emergency','dx10','dx100','dx101','dx102','dx103','dx104','dx105','dx106','dx107','dx108','dx109','dx11','dx110','dx111','dx112','dx113','dx114','dx115','dx116','dx117','dx118','dx119','dx12','dx120','dx121','dx122','dx123','dx124','dx125','dx126','dx127','dx128','dx129','dx13','dx130','dx131','dx132','dx133','dx134','dx135','dx136','dx137','dx138','dx139','dx14','dx140','dx141','dx142','dx143','dx144','dx145','dx146','dx147','dx148','dx149','dx15','dx151','dx152','dx153','dx154','dx155','dx156','dx157','dx158','dx159','dx16','dx160','dx161','dx162','dx163','dx164','dx165','dx166','dx168','dx17','dx170','dx171','dx173','dx175','dx178','dx18','dx181','dx19','dx195','dx197','dx198','dx199','dx2','dx200','dx201','dx202','dx203','dx204','dx205','dx206','dx207','dx208','dx209','dx210','dx211','dx212','dx213','dx215','dx217','dx22','dx224','dx225','dx226','dx228','dx229','dx23','dx230','dx231','dx232','dx233','dx234','dx235','dx236','dx237','dx238','dx239','dx24','dx240','dx241','dx242','dx243','dx244','dx245','dx246','dx247','dx248','dx249','dx25','dx250','dx251','dx252','dx253','dx255','dx256','dx257','dx259','dx26','dx2603','dx2607','dx2608','dx2613','dx2615','dx2616','dx2617','dx2618','dx2619','dx2620','dx2621','dx27','dx29','dx3','dx32','dx33','dx35','dx36','dx37','dx38','dx39','dx4','dx40','dx42','dx43','dx44','dx45','dx46','dx47','dx48','dx49','dx5','dx50','dx51','dx52','dx53','dx54','dx55','dx57','dx58','dx59','dx6','dx60','dx62','dx63','dx64','dx650','dx651','dx652','dx653','dx654','dx655','dx657','dx658','dx659','dx660','dx661','dx662','dx663','dx670','dx7','dx76','dx77','dx78','dx79','dx8','dx80','dx81','dx82','dx83','dx84','dx85','dx86','dx87','dx88','dx89','dx90','dx91','dx92','dx93','dx94','dx95','dx96','dx97','dx98','dx99']

# lab_names = ['Bicarbonate','Calcium','Creatinine','Hematocrit','Magnesium','Potassium','Sodium','Urea Nitrogen','pH']
lab_names = ['Albumin','Ammonia','Amylase','AsparateAminotransferaseAST','Bicarbonate','Chloride','Creatinine','Globulin','Glucose','Hematocrit','Hemoglobin','Magnesium','Potassium','Protein','Sodium','Temperature','Triglycerides','UreaNitrogen','WBC','pCO2','pH','pO2']

fc_output_vector_size = 24
rnn_output_size = 24
w2_size = 6
rnn_num_hidden = 32
maxSeqLength = 5
learning_rate = 0.0001
num_batches = 7001
if os.path.exists('train/'):
    shutil.rmtree('train/')
if os.path.exists('validation/'):
    shutil.rmtree('validation/')
train()

print(fc_output_vector_size)
print(rnn_output_size)
print(w2_size)
print(rnn_num_hidden)
print(maxSeqLength)
print(learning_rate)
print(num_batches)
