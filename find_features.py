import sys
import os
sys.path.append(os.path.abspath("../"))
from models.mm_embeddings import AudioTextVideoEmbedding
import numpy as np
import haiku as hk
import jax.numpy as jnp
import jax
import functools
from mmv.utils import checkpoint
from absl import flags, app
from mmv import config
from typing import Any, Dict
import tensorflow as tf
import skvideo.io
import pickle as pkl

flags.DEFINE_string('checkpoint_path', './mmv_tsm_resnet_x2.pkl',
                    'The directory to load pre-trained weights from.')
flags.DEFINE_string('dataset_folder', '/home2/dhawals1939/bullu/repos/video/',
                    'The directory with all the videos.')
flags.DEFINE_string('output_folder', '../outputs/',
                    'The directory used as output folder.')

FLAGS = flags.FLAGS

def forward_fn(images: jnp.ndarray, is_training: bool, final_endpoint: str, model_config: Dict[str, Any]):
    module = AudioTextVideoEmbedding(**model_config, word_embedding_matrix = None)
    """
    Args:
        images: A 5-D float array of shape `[B, T, H, W, 3]`.
        is_training: Whether to use training mode.
        final_endpoint: Up to which endpoint to run / return.
    """
    return module(images=images, 
                  is_training=is_training, 
                  final_endpoint=final_endpoint,
                  audio_spectrogram=None,
                  word_ids=None)['vid_repr']

def main(argv):
    os.makedirs(FLAGS.output_folder, exist_ok=True)

    del argv
    model_config = config.get_model_config(FLAGS.checkpoint_path)

    pretrained_weights = checkpoint.load_checkpoint(FLAGS.checkpoint_path)
    params = pretrained_weights['params']
    state = pretrained_weights['state']
    
    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
    forward_apply = jax.jit(functools.partial(forward.apply, is_training=False, final_endpoint='Embeddings', model_config=model_config, params=params, state=state))

    for path in sorted(tf.io.gfile.listdir(FLAGS.dataset_folder)):
        video_name = os.path.splitext(path)[0]
        print(f"Finding the representation of {path}")
        if os.path.exists(os.path.join(FLAGS.output_folder, video_name + ".pkl")):
            print(f"Representation of {path} already exists")
            continue

        video_data = skvideo.io.vread(os.path.join(FLAGS.dataset_folder, path))
        video_data = (video_data[np.newaxis, ...]).astype(np.float32) / 255.0
        # print(video_data.shape, video_data.dtype) # (1, #frames, H, W, 3) float32

        vid_representation_test, _ = forward_apply(images=video_data)
        for v in vid_representation_test:
            print(v)
        # with open(os.path.join(FLAGS.output_folder, video_name + ".pkl"), "wb") as f:
        #     pkl.dump(vid_representation_test, f)
        exit(0)

if __name__ == '__main__':
    app.run(main)