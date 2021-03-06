import keys
import tweepy
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import os
import re
import pandas as pd
from datetime import datetime

class Model:
    
    OUTPUT_DIR = '~/Twitter-Clustering/output'
    USE_BUCKET = True
    DO_DELETE = False
    USE_BUCKET = False
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 10.0
    WARMUP_PROPORTION = 0.1
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    df = pd.read_csv('training.csv', encoding='latin-1')
    df = df.drop(df.columns[[1,2,3,4]], axis=1)
    df.columns = ["label", "text"]
    df = df[["text", "label"]]
    train = df.sample(10000)
    test = df.sample(5000)
    train.columns

    DATA_COLUMN = 'text'
    LABEL_COLUMN = 'label'
    label_list = [0, 4]
    
    # train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid = None, text_a = x[DATA_COLUMN], text_b = None, label = x[LABEL_COLUMN]), axis = 1)
    # test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid = None, text_a = x[DATA_COLUMN], text_b = None, label = x[LABEL_COLUMN]), axis = 1)

    def create_tokenizer_from_hub_module(self):
        with tf.Graph().as_default():
            bert_module = hub.Module(self.BERT_MODEL_HUB)
            tokenization_info = bert_module(signature = "tokenization_info", as_dict = True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])

        return bert.tokenization.FullTokenizer(vocab_file = vocab_file, do_lower_case = do_lower_case)


    MAX_SEQ_LENGTH = 128

    def model_fn_builder(self, num_labels, learning_rate, num_train_steps, num_warmup_steps):
        def model_fn(features, labels, mode, params):
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
            if not is_predicting:
                (loss, predicted_labels, log_probs) = self.create_model( is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

                def metric_fn(label_ids, predicted_labels):
                    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                    f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                    auc = tf.metrics.auc(label_ids, predicted_labels)
                    recall = tf.metrics.recall(label_ids, predicted_labels)
                    precision = tf.metrics.precision(label_ids, predicted_labels) 
                    true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
                    true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)   
                    false_pos = tf.metrics.false_positives(label_ids, predicted_labels)  
                    false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)
                    return {
                        "eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "true_positives": true_pos,
                        "true_negatives": true_neg,
                        "false_positives": false_pos,
                        "false_negatives": false_neg
                    }
                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

                else:
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
            else:
                (predicted_labels, log_probs) = self.create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
                predictions = {
                    'probabilities': log_probs,
                    'labels': predicted_labels
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        return model_fn
    
    def __init__(self): 
        self.train_InputExamples = self.train.apply(lambda x: bert.run_classifier.InputExample(guid = None, text_a = x[self.DATA_COLUMN], text_b = None, label = x[self.LABEL_COLUMN]), axis = 1)
        self.test_InputExamples = self.test.apply(lambda x: bert.run_classifier.InputExample(guid = None, text_a = x[self.DATA_COLUMN], text_b = None, label = x[self.LABEL_COLUMN]), axis = 1)
        self.tokenizer = self.create_tokenizer_from_hub_module()
        self.train_features = bert.run_classifier.convert_examples_to_features(self.train_InputExamples, self.label_list, self.MAX_SEQ_LENGTH, self.tokenizer)
        self.test_features = bert.run_classifier.convert_examples_to_features(self.test_InputExamples, self.label_list, self.MAX_SEQ_LENGTH, self.tokenizer)
        self.num_train_steps = int(len(self.train_features) / self.BATCH_SIZE * self.NUM_TRAIN_EPOCHS)
        self.num_warmup_steps = int(self.num_train_steps * self.WARMUP_PROPORTION)
        self.model_fn = self.model_fn_builder(
            num_labels=len(Model.label_list),
            learning_rate=self.LEARNING_RATE,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps)
        self.run_config = tf.estimator.RunConfig(
            model_dir=self.OUTPUT_DIR,
            save_summary_steps=self.SAVE_SUMMARY_STEPS,
            save_checkpoints_steps=self.SAVE_CHECKPOINTS_STEPS)
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=self.run_config,
            params={"batch_size": self.BATCH_SIZE})   
        tf.gfile.MakeDirs(self.OUTPUT_DIR)
        
    
    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
        bert_module = hub.Module(self.BERT_MODEL_HUB, trainable = True)
        bert_inputs = dict(input_ids = input_ids, input_mask = input_mask, segment_ids = segment_ids)
        bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)
        output_layer = bert_outputs["pooled_output"]
        hidden_size = output_layer.shape[-1].value
        output_weights = tf.get_variable("output_weights", [num_labels, hidden_size], initializer = tf.truncated_normal_initializer(stddev = .02))
        output_bias = tf.get_variable("output_bias", [num_labels], initializer = tf.zeros_initializer())
        with tf.variable_scope("loss"):
            output_layer = tf.nn.dropout(output_layer, keep_prob = .9)
            logits = tf.matmul(output_layer, output_weights, transpose_b = True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        if is_predicting:
            return (predicted_labels, log_probs)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)




    def getPrediction(self, in_sentences):
        self.labels = self.label_list
        self.input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
        self.input_features = run_classifier.convert_examples_to_features(self.input_examples, self.label_list, self.MAX_SEQ_LENGTH, self.tokenizer)
        self.predict_input_fn = run_classifier.input_fn_builder(features=self.input_features, seq_length=self.MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
        self.predictions = self.estimator.predict(self.predict_input_fn)
        return [(sentence, prediction['probabilities'], self.labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, self.predictions)]

