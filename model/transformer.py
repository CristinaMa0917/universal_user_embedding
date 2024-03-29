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


class _AveragePooling:
    def __init__(self, **kwargs):
        pass

    def __call__(self, embeddings, masks):
        # batch * length * 1
        multiplier = tf.expand_dims(masks, axis=-1)
        embeddings_sum = tf.reduce_sum(
            tf.multiply(multiplier, embeddings),
            axis=2
        )
        length = tf.expand_dims(tf.maximum(tf.reduce_sum(masks, axis=2), 1.0), axis=-1)

        embedding_avg = embeddings_sum / length
        return embedding_avg


class _WordEmbeddingEncoder:
    def __init__(self, config, word_count, dimension, training, scope_name, date_span, enable_date_time_emb, *args, **kwargs):
        self.config = config
        self._scope_name = scope_name
        self._enable_date_time_emb = enable_date_time_emb
        self._word_dimension = dimension
        self._average_pooling = _AveragePooling()
        self.training = training
        self._initialized = False
        self.que_len = self.config["finetune_query_length"]
        self.seq_len = self.config["finetune_seq_length"]
        self.init_ckt = self.config["init_checkpoint"]
        print("***********************%s init_ckt : "%self.init_ckt)

        with tf.variable_scope(scope_name, reuse=False):
            self._word_embedding = tf.get_variable(
                "WordEmbedding",
                [word_count + 1, dimension],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=training
            )
            # zeros padding
            self._word_embedding = tf.concat((tf.zeros(shape=[1, dimension]),self._word_embedding[1:, :]), 0)

            if self._enable_date_time_emb:
                self._date_embedding = tf.get_variable(
                    "TimeDiffEmbedding",
                    [date_span + 1, int(self._word_embedding.shape[1])],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=training
                )
                self._date_embedding = tf.concat((tf.zeros(shape=[1, dimension]), self._date_embedding[1:, :]), 0)



    def __call__(self, content_words, content_len, date):
        """
        获得句子的embedding
        :param tokens: batch_size * max_seq_len
        :param masks: batch_size * max_seq_len
        :return:
        """
        word_embedding_ave = tf.nn.embedding_lookup(self._word_embedding, content_words)
        # word_embedding shape(Batch_size, query_counts, query_length, word_dimension)
        word_embedding_ave = tf.reshape(word_embedding_ave, shape=(-1, 300, 8, self._word_dimension))
        # content_len shape(Batch_size, query_counts, 8(mask_dim))
        content_len = tf.reshape(content_len, shape=(-1, 300, 8))
        # avg_embedding shape(Batch_size, query_counts, word_dimension)
        word_embedding_ave = self._average_pooling(word_embedding_ave, content_len)
        print("avg_embedding shape is ", word_embedding_ave.shape) #[b, q, h]

        content_words_reshape = tf.reshape(content_words,[-1, self.seq_len]) #[b*q, s]
        content_len_reshape = tf.reshape(content_len, [-1, self.seq_len]) #[b*q, s]
        pooled_mask = tf.reshape(tf.reduce_max(content_len_reshape, -1), [-1, self.que_len])  # [b, q]

        model = bert_base.BertModel(
            config=self.config,
            is_training=self.training,
            input_ids=content_words_reshape,
            input_mask=content_len_reshape,
            use_one_hot_embeddings=self.config["use_one_hot_embeddings"])

        if not self._initialized:
            tf.logging.info('================== init pretrain bert from checkpoint ============')
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            if self.init_ckt:
                (assignment_map, initialized_variable_names) = \
                    bert_base.get_assignment_map_from_checkpoint(tvars, self.init_ckt)
                tf.train.init_from_checkpoint(self.init_ckt, assignment_map)

            tf.logging.info("**** initialized_variable_names  **** \n %s"%initialized_variable_names)
            tf.logging.info("**** Trainable Variables ****")

            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)
            self._initialized = True

        word_embedding_bert = model.get_sequence_output() # [b*q, s, h]
        word_embedding_bert = ave_pooling(word_embedding_bert, content_len_reshape) #[b*q, s]
        word_embedding_bert = tf.reshape(word_embedding_bert, [-1, self.que_len, 96]) # [b,q, 96]

        word_embedding = tf.concat([word_embedding_bert,word_embedding_ave], axis=-1)
        word_embedding = layers.fully_connected(word_embedding, self._word_dimension, activation_fn=None,
                                                scope='word_emb_hidden_layer')
        word_embedding = tf.multiply(tf.expand_dims(pooled_mask,-1), word_embedding)


        if self._enable_date_time_emb:
            date = tf.reshape(date, shape=(-1, 300, 8))
            time_diff = tf.reshape(date[:, :, 0], shape=(-1, 300))
            time_embedding = tf.nn.embedding_lookup(self._date_embedding, time_diff)
            embedding = word_embedding + time_embedding
            print("Date embedding regularization is enabled")
            distinct_times = tf.unique(tf.reshape(time_diff, [-1]))[0]
            distinct_time_embedding = tf.nn.embedding_lookup(self._date_embedding, distinct_times)
            return embedding, 0.001 * tf.nn.l2_loss(distinct_time_embedding)
        else:
            return word_embedding, 0

class _Transformer:
    def __init__(self, encoder, dropout_rate, batch_size, training, scope_name, num_blocks, num_heads, num_vocabulary,
                 feed_forward_in_dim, model_dim):
        self._scope = tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE)
        self._encoder = encoder
        self._batch_size = batch_size
        self.num_blocks = num_blocks
        self.training = training
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.num_vocabulary = num_vocabulary
        self.feed_forward_in_dim = feed_forward_in_dim
        self.model_dim = model_dim

    def __call__(self, content_words, content_len, date, target, target_len):
        embeddings, reg_loss = self._encoder(content_words, content_len, date)

        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                embeddings = multihead_attention(queries=embeddings,
                                                 keys=embeddings,
                                                 values=embeddings,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 training=self.training,
                                                 causality=False)
                # feed forward
                embeddings = ff(embeddings, num_units=[self.feed_forward_in_dim, self.model_dim])
        # shape(?,300,512)
        outputs = tf.reduce_max(embeddings, axis=1)
        output_feature = outputs
        if self.training and self.dropout_rate > 0:
            print("In training mode, use dropout")
            outputs = tf.nn.dropout(outputs, keep_prob=1 - self.dropout_rate)

        with tf.variable_scope("MlpLayer") as hidden_layer_scope:
            outputs = layers.fully_connected(
                outputs, num_outputs=self.model_dim, activation_fn=tf.nn.tanh,
                scope=hidden_layer_scope, reuse=tf.AUTO_REUSE
            )
        outputs = layers.linear(
            outputs, self.num_vocabulary, scope="Logit_layer", reuse=tf.AUTO_REUSE
        )

        loss = None
        training_list = []
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
                    dense_shape=[self._batch_size, self.num_vocabulary]
                )
            )
            target_dist = expanded_target / tf.cast(tf.reshape(target_len, [-1, 1]), tf.float32)

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=outputs,
                    labels=tf.stop_gradient(target_dist)
                ) + reg_loss
            )

            # set differnt lr
            print("**** learning_rate ****")
            tvars = tf.trainable_variables()
            for var in tvars:
                if not var.op.name.startswith("bert"):
                    training_list.append(var)


        return Namespace(
            logit=outputs,
            feature=output_feature,
            loss=loss,
            training_list=training_list
        )


class TextTransformerNet:
    ModelConfigs = namedtuple("ModelConfigs", ("dropout_rate", "num_vocabulary",
                                               "feed_forward_in_dim", "model_dim",
                                               "num_blocks", "num_heads", "enable_date_time_emb",
                                               "word_emb_dim", "date_span"))

    def __init__(self, bert_config, model_configs, train_configs, predict_configs, run_configs):
        self.config = bert_config
        self._model_configs = model_configs
        self._train_configs = train_configs
        self._predict_configs = predict_configs
        self._run_configs = run_configs

    def _train(self, model_output, labels):
        loss = model_output.loss
        training_list = model_output.training_list
        train_op = tf.train.AdamOptimizer(learning_rate=self.config["finetune_lr"]).minimize(loss,
                                                                                        global_step=tf.train.get_global_step(),
                                                                                        var_list=training_list)

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
        word_encoder = _WordEmbeddingEncoder(
            config=self.config,
            scope_name="encoder",
            word_count=self._model_configs.num_vocabulary,
            dimension=self._model_configs.word_emb_dim,
            date_span=self._model_configs.date_span,
            training=training,
            enable_date_time_emb=self._model_configs.enable_date_time_emb
        )

        transformer_enc = _Transformer(encoder=word_encoder,
                                       dropout_rate=self._train_configs.dropout_rate,
                                       batch_size=self._train_configs.batch_size,
                                       training=training,
                                       scope_name="Transformer",
                                       num_blocks=self._model_configs.num_blocks,
                                       num_heads=self._model_configs.num_heads,
                                       num_vocabulary=self._model_configs.num_vocabulary,
                                       feed_forward_in_dim=self._model_configs.feed_forward_in_dim,
                                       model_dim=self._model_configs.model_dim)

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