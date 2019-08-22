# -*- coding: utf-8 -*-#
import tensorflow as tf
import argparse
import model
from data_loader import loader
from util import helper
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
from configs.config import parse_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--buckets", type=str, help="Worker task index")
    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--max_length", type=int, default=2400)
    parser.add_argument("--target_length", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--init_ckt_dir", type=str)
    parser.add_argument("--init_ckt_step", type=int)
    return parser.parse_known_args()[0]


def main():
    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Config
    config = parse_config('MiniBERT')
    config["init_checkpoint"] = args.buckets + args.init_ckt_dir + "/model.ckpt-{}".format(args.init_ckt_step)


    # Check if the model has already exisited
    model_save_dir = args.buckets + args.checkpoint_dir
    if tf.gfile.Exists(model_save_dir + "/checkpoint"):
        raise ValueError("Model %s has already existed, please delete them and retry" % model_save_dir)

    helper.dump_args(model_save_dir, args)

    transformer_model = model.TextTransformerNet(
        bert_config=config,
        train_configs=model.TrainConfigs(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            dropout_rate = args.dropout_rate
        ),
        predict_configs=None,
        run_configs=model.RunConfigs(
            log_every=50
        )
    )
    # checkpoint_path = None
    # if args.step > 0:
    #     checkpoint_path = model_save_dir + "/model.ckpt-{}".format(args.step)
    # warm_start_settings = tf.estimator.WarmStartSettings(checkpoint_path,
    #                                                      vars_to_warm_start='(.*Embedding|Conv-[1-4]|MlpLayer-1)')
    cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps(
        'nccl'
    )
    distribution = tf.contrib.distribute.MirroredStrategy(
        num_gpus=4, cross_tower_ops=cross_tower_ops,
        all_dense=False
    )

    estimator = tf.estimator.Estimator(
        model_fn=transformer_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False),
                allow_soft_placement=True
            ),
            save_checkpoints_steps=args.snapshot,
            keep_checkpoint_max=20,
            train_distribute=distribution
        )
    )
    print("Start training......")
    tf.estimator.train(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=loader.OdpsDataLoader(
                table_name=args.tables,
                mode=tf.estimator.ModeKeys.TRAIN,
                hist_length=args.max_length,
                target_length=args.target_length,
                batch_size=args.batch_size
            ).input_fn,
            max_steps=args.max_steps
        )
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

