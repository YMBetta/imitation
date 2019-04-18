
import numpy as np

discriminator_update = 1

batch_size = 1000

learning_rate = 1e-5  # lr of discriminator, should smaller than lr of policy.

beta1 = 0.5

beta2 = 0.9

epsilon = 1e-8

discriminator_params = {
    'layer_001': {
      'nh': 16,
      'init_scale': np.sqrt(2),
      'init_bias': 0.0,
      'layer_norm': False,
    },
    'layer_002': {
      'nh': 32,
      'init_scale': np.sqrt(2),
      'init_bias': 0.0,
      'layer_norm': False,
    },
    'layer_003': {
      'nh': 16,
      'init_scale': np.sqrt(2),
      'init_bias': 0.0,
      'layer_norm': False,
    },
    'layer_004': {
      'nh': 1,
      'init_scale': np.sqrt(2),
      'init_bias': 0.0,
      'layer_norm': False,
    }
}