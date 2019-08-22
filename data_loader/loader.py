# -*- coding: utf-8 -*-#
import tensorflow as tf


class OdpsDataLoader:
    def __init__(self, table_name, hist_length, target_length, mode, repeat=None, batch_size=128, shuffle=2000, slice_id=0, slice_count=1,query_counts = 300):
        # Avoid destroying input parameter
        self._table_name = table_name
        self._max_length = hist_length
        self._target_length = target_length
        self._slice_id = slice_id
        self._slice_count = slice_count
        self._batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._mode = mode
        self._query_counts = query_counts

    def _text_content_parser(self, text, max_length):
        word_strs = tf.string_split([text], " ")
        return tf.string_to_number(word_strs.values, out_type=tf.int64)[:max_length], tf.minimum(tf.shape(word_strs)[-1], max_length)

    def _train_data_parser(self, oneid, content, content_count, target):
        content_words, _ = self._text_content_parser(content, self._max_length)
        target_words, target_len = self._text_content_parser(target, self._target_length)
        words = tf.div(content_words, 256) + 1
        content_len = tf.cast(tf.not_equal(content_words, 0), tf.float32)
        dates = tf.bitwise.bitwise_and(content_words, 255)
        return {
            "oneid": oneid,
            "content_words": words,
            "content_len": content_len,
            "date": dates,
            "target_words": target_words,
            "target_len": target_len
        }, tf.constant(0, dtype=tf.int32) # fake label


    def _test_data_parser(self, oneid, content):
        content_words, _ = self._text_content_parser(content, self._max_length)
        words = tf.div(content_words, 256) + 1
        content_len = tf.cast(tf.not_equal(content_words, 0), tf.float32)
        dates = tf.bitwise.bitwise_and(content_words, 255)
        return {
            "oneid": oneid,
            "content_words": words,
            "content_len": content_len,
            "date": dates
        }, tf.constant(0, dtype=tf.int32) # fake label


    def _train_data_fn(self):
        with tf.device("/cpu:0"):
            # input format: oneid, content, content_count, target
            dataset = tf.data.TableRecordDataset(
                self._table_name,
                record_defaults=["", "", "", ""],
                slice_id=self._slice_id,
                slice_count=self._slice_count
            )

            dataset = dataset.map(self._train_data_parser, num_parallel_calls=4)
            if self._shuffle > 0:
                dataset = dataset.shuffle(self._shuffle)

            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            dataset = dataset.prefetch(40000)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "oneid": [],
                        "content_words": [self._max_length],
                        "date":[self._max_length],
                        "content_len": [self._max_length],
                        "target_words": [self._target_length],
                        "target_len": [],
                    }, [])
            )
            return dataset

    def _test_data_fn(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TableRecordDataset(
                self._table_name,
                record_defaults=["", ""],
                slice_id=self._slice_id,
                slice_count=self._slice_count
            )

            dataset = dataset.map(self._test_data_parser, num_parallel_calls=4)
            if self._shuffle > 0:
                dataset = dataset.shuffle(self._shuffle)

            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            #dataset = dataset.prefetch(40000)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "oneid": [],
                        "content_words": [self._max_length],
                        "date":[self._max_length],
                        "content_len": [self._max_length]
                    }, [])
            )

            return dataset

    def input_fn(self):
        return self._train_data_fn() if self._mode is tf.estimator.ModeKeys.TRAIN else self._test_data_fn()



class LocalFileDataLoader:
    def __init__(self, file_path, hist_length, target_length, mode, repeat=None, maxlength = 2400, batch_size=128, shuffle=2000):
        # Avoid destroying input parameter
        self._file_path = file_path
        self._hist_length = hist_length
        self._target_length = target_length
        self._batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._mode = mode
        self._max_length = maxlength
    def _text_content_parser(self, text, max_length):
        word_strs = tf.string_split([text], " ")
        return tf.string_to_number(word_strs.values, out_type=tf.int64)[:max_length], tf.minimum(tf.shape(word_strs)[-1], max_length)

    def _train_data_parser(self, line):
        values = tf.string_split(tf.reshape(line, [-1]), ",").values
        oneid, content, content_counts, target = values[0], values[1], values[2], values[3]
        content_words, _ = self._text_content_parser(content, self._max_length)
        target_words, target_len = self._text_content_parser(target, self._target_length)
        words = tf.div(content_words, 256) + 1
        content_len = tf.cast(tf.not_equal(content_words, 0), tf.float32)
        dates = tf.bitwise.bitwise_and(content_words, 255)
        return {
            "oneid": oneid,
            "content_words": words,
            "content_len": content_len,
            "date": dates,
            "target_words": target_words,
            "target_len": target_len
        }, tf.constant(0, dtype=tf.int32) # fake label

    def _test_data_parser(self, line):
        values = tf.string_split(tf.reshape(line, [-1]), ",").values
        oneid, hist = values[0], values[1]
        hist_words, _ = self._text_content_parser(hist, self._hist_length)
        return {
            "oneid": oneid,
            "hist_words": hist_words,
        }, tf.constant(0, dtype=tf.int32)  # fake label

    def _train_data_fn(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TextLineDataset(self._file_path)
            dataset = dataset.map(self._train_data_parser, num_parallel_calls=4)
            if self._shuffle > 0:
                dataset = dataset.shuffle(self._shuffle)

            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            dataset = dataset.prefetch(40000)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "oneid": [],
                        "content_words": [self._max_length],
                        "date":[self._max_length],
                        "content_len": [self._max_length],
                        "target_words": [self._target_length],
                        "target_len": [],
                    }, [])
            )

            return dataset

    def _test_data_fn(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TextLineDataset(self._file_path)
            dataset = dataset.map(self._test_data_parser, num_parallel_calls=4)
            if self._shuffle > 0:
                dataset = dataset.shuffle(self._shuffle)

            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            dataset = dataset.prefetch(40000)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "oneid": [],
                        "hist_words": [self._hist_length],
                    }, [])
            )

            return dataset

    def input_fn(self):
        return self._train_data_fn() if self._mode is tf.estimator.ModeKeys.TRAIN else self._test_data_fn()
