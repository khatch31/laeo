from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from optax._src import utils

def sigmoid_positive_unlabeled_loss(logits, labels, eta=0.5):
  chex.assert_type([logits], float)
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  # BCE loss is: return labels * log_p - (1. - labels) * log_not_p
  # return (-eta * labels * log_p) - ((1. - labels) * log_not_p) + (eta * labels + log_not_p)
  return (-eta * labels * log_p) - ((1. - labels) * log_not_p) + (eta * labels * log_not_p)


  # bce = -target * math_ops.log(output) - (1 - target) * math_ops.log(1 - output))
  # pu = -eta * target * math_ops.log(output) - (1 - target) * math_ops.log(1 - output) + eta * target * math_ops.log(1 - output)

  # bce = target * math_ops.log(output + epsilon())
  # bce += (1 - target) * math_ops.log(1 - output + epsilon())
  # return -bce
