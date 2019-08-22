# -*- coding: UTF-8 -*-
from collections import namedtuple
from argparse import Namespace
import tensorflow as tf
from tensorflow.contrib import layers
from util import diagnose
from model.module import  ff, multihead_attention
from model import bert_base


try:
    SparseTensor = tf.sparse.SparseTensor
    to_dense = tf.sparse.to_dense
except:
    SparseTensor = tf.SparseTensor
    to_dense = tf.sparse_tensor_to_dense


class _Transformer:
    def __init__(self, config, batch_size, training, scope_name):
        self._config = config
        self._scope_name = scope_name
        self._training = training
        self._initialized = False
        self._que_len = config["finetune_query_length"]
        self._seq_len = config["finetune_seq_length"]
        self._init_ckt = config["init_checkpoint"]
        self._word_dimension = config["word_emb_dim"]
        self._num_blocks = config["num_blocks"]
        self._dropout_rate =config["dropout_rate"]
        self._num_heads = config["num_heads"]
        self._transformer_vocabulary = config["transformer_vocabulary"]+1
        self._feed_forward_in_dim = config["feed_forward_in_dim"]
        self._model_dim = config["model_dim"]
        self._scope = tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE)
        self._batch_size = batch_size

        print("***********************%s init_ckt : " % self._init_ckt)
        if config["enable_date_time_emb"]:
            self._date_embedding = tf.get_variable(
                "TimeDiffEmbedding",
                [config["date_span"] + 1, self._word_dimension],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=training
            )
            self._date_embedding = tf.concat((tf.zeros(shape=[1, self._word_dimension]), self._date_embedding[1:, :]),
                                             0)

    def __call__(self, content_words, content_len, date, target, target_len):
        # bert
        content_words_reshape = tf.reshape(content_words, [-1, self._seq_len])  # [b*q,s]
        content_len_reshape = tf.reshape(content_len, [-1, self._seq_len])  # [b*q, s]
        pooled_mask = tf.reshape(tf.reduce_max(content_len_reshape, -1), [-1, self._que_len])  # [b, q]

        model = bert_base.BertModel(
            config=self._config,
            is_training=self._training,
            input_ids=content_words_reshape,
            input_mask=content_len_reshape,
            use_one_hot_embeddings=self._config["use_one_hot_embeddings"])

        word_embedding = model.get_sequence_output()  # [b*q, s,h]
        word_embedding = max_pooling(word_embedding, content_len_reshape)  # [b*q,h]
        word_embedding = layers.fully_connected(word_embedding, self._word_dimension, activation_fn=None,
                                                scope='word_emb_hidden_layer')
        word_embedding = tf.reshape(word_embedding, [-1, self._que_len, self._word_dimension])
        word_embedding = tf.multiply(tf.expand_dims(pooled_mask,-1), word_embedding)

        date = tf.reshape(date, shape=(-1, 300, 8))
        time_diff = tf.reshape(date[:, :, 0], shape=(-1, 300))
        time_embedding = tf.nn.embedding_lookup(self._date_embedding, time_diff)
        embeddings = word_embedding + time_embedding

        print("Date embedding regularization is enabled")
        distinct_times = tf.unique(tf.reshape(time_diff, [-1]))[0]
        distinct_time_embedding = tf.nn.embedding_lookup(self._date_embedding, distinct_times)
        reg_loss= 0.001 * tf.nn.l2_loss(distinct_time_embedding)

        with self._scope:
            # transformer
            for i in range(self._num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    embeddings = multihead_attention(queries=embeddings,
                                                     keys=embeddings,
                                                     values=embeddings,
                                                     num_heads=self._num_heads,
                                                     dropout_rate=self._dropout_rate,
                                                     training=self._training,
                                                     causality=False)
                    # feed forward
                    embeddings = ff(embeddings, num_units=[self._feed_forward_in_dim, self._model_dim])

        # shape(?,300,512)
        outputs = tf.reduce_max(embeddings, axis=1)
        output_feature = outputs
        if self._training and self._dropout_rate > 0:
            print("In training mode, use dropout")
            outputs = tf.nn.dropout(outputs, keep_prob=1 - self._dropout_rate)

        with tf.variable_scope("MlpLayer") as hidden_layer_scope:
            outputs = layers.fully_connected(
                outputs, num_outputs=self._model_dim, activation_fn=tf.nn.tanh,
                scope=hidden_layer_scope, reuse=tf.AUTO_REUSE
            )
        outputs = layers.linear(
            outputs, self._transformer_vocabulary, scope="Logit_layer", reuse=tf.AUTO_REUSE
        )

        loss = None
        self.learning_rate = dict()
        if target is not None:
            non_zero_indices = tf.where(tf.not_equal(target, 0))
            col_indices = tf.cast(tf.gather_nd(target, non_zero_indices), tf.int64)
            expanded_target = to_dense(
                SparseTensor(
                    indices=tf.concat([
                        tf.reshape(non_zero_indices[:, 0], [-1, 1]),
                        tf.reshape(col_indices, [-1, 1]),
                    ], axis=1),
                    values=tf.ones([tf.shape(non_zero_indices)[0]], dtype=tf.float32),
                    dense_shape=[self._batch_size, self._transformer_vocabulary]
                )
            )
            target_dist = expanded_target / tf.cast(tf.reshape(target_len, [-1, 1]), tf.float32)

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=outputs,
                    labels=tf.stop_gradient(target_dist)
                ) + reg_loss
            )

            # init from ckt
            if not self._initialized:
                tvars = tf.trainable_variables()
                tf.logging.info('================== init pretrain bert from checkpoint%s ============'%self._init_ckt)
                initialized_variable_names = {}
                if self._init_ckt:
                    (assignment_map, initialized_variable_names) = \
                        bert_base.get_assignment_map_from_checkpoint(tvars, self._init_ckt)
                    tf.train.init_from_checkpoint(self._init_ckt, assignment_map)

                tf.logging.info("**** initialized_variable_names  **** \n %s" % initialized_variable_names)
                tf.logging.info("**** Trainable Variables ****")

                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)
                self._initialized = True

            # set differnt lr
            print("**** learning_rate ****")
            tvars = tf.trainable_variables()
            for var in tvars:
                if var.op.name.startswith("bert"):
                    self.learning_rate[var.op.name] = 0.05 #5e-5/1e-3
                else:
                    self.learning_rate[var.op.name] = 1
                print(var.op.name, ': ', self.learning_rate[var.op.name])

        return Namespace(
            logit=outputs,
            feature=output_feature,
            loss=loss,
            lr=self.learning_rate
        )


class TextTransformerNet:

    def __init__(self, bert_config, train_configs, predict_configs, run_configs):
        self.config = bert_config
        self._train_configs = train_configs
        self._predict_configs = predict_configs
        self._run_configs = run_configs

    def _train(self, model_output, labels):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config["finetune_lr"])
        grads_and_vars = optimizer.compute_gradients(model_output.loss)
        grads_and_vars_mult = []
        lr = model_output.lr

        for grad, var in grads_and_vars:
            try:
                grad *= lr[var.op.name]
            except:
                print("Pass Set %s" % var.op.name)
                pass
            grads_and_vars_mult.append((grad, var))
        train_op = optimizer.apply_gradients(grads_and_vars_mult, tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=model_output.loss,
            train_op=train_op,
            training_hooks=[
                tf.train.LoggingTensorHook(
                    {
                        "loss": model_output.loss,
                        # "accuracy": 100. * tf.reduce_mean(tf.cast(tf.equal(tf.cast(model_output.prediction,tf.int32), labels), tf.float32))
                    },
                    every_n_iter=self._run_configs.log_every
                ),
                diagnose.GraphPrinterHook()#,
                # tf.train.ProfilerHook(
                #     save_secs=60,
                #     show_memory=True,
                #     output_dir="oss://wf135777-lab/profile/"
                # )
            ]
        )

    def _predict(self, model_output):
        outputs = dict(
            oneid=model_output.oneid,
            feature=tf.reduce_join(
                tf.as_string(model_output.feature, shortest=True, scientific=False),
                axis=1,
                separator=self._predict_configs.separator
            )
        )
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=outputs
        )

    def _build_model(self, features, labels, mode):
        training = mode is tf.estimator.ModeKeys.TRAIN

        transformer_enc = _Transformer(
                                       config=self.config,
                                       batch_size=self._train_configs.batch_size,
                                       training=training,
                                       scope_name="Transformer",
                                       )

        oneid = features["oneid"]
        model_output = transformer_enc(
            content_words=features["content_words"],
            content_len=features["content_len"],
            date=features["date"],
            target=features.get("target_words"),
            target_len=features.get("target_len")
        )
        model_output.oneid = oneid
        return model_output

    def model_fn(self, features, labels, mode):
        model_output = self._build_model(features, labels, mode)
        if mode is tf.estimator.ModeKeys.TRAIN:
            return self._train(model_output, labels)
        elif mode is tf.estimator.ModeKeys.PREDICT:
            return self._predict(model_output)
        elif mode is tf.estimator.ModeKeys.EVAL:
            return self._evaluate(model_output, labels)

def ave_pooling(embeddings, masks):
    # batch * length * 1
    multiplier = tf.expand_dims(masks, axis=-1)
    embeddings_sum = tf.reduce_sum(tf.multiply(multiplier, embeddings),axis=1)
    length = tf.expand_dims(tf.maximum(tf.reduce_sum(masks, axis=1), 1.0), axis=-1)
    embedding_avg = embeddings_sum / length
    return embedding_avg

def max_pooling(embeddings, masks):
    # batch * length * 1
    multiplier = tf.expand_dims(masks, axis=-1)
    embeddings_max = tf.reduce_max(tf.multiply(multiplier, embeddings),axis=1)
    return embeddings_max