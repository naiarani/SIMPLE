import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Add, Dense, Lambda
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.distributions import CategoricalProbabilityDistribution

# Define constants
ACTIONS = 25  # Number of actions in Calico game
FEATURE_SIZE = 128
DEPTH = 5
VALUE_DEPTH = 1
POLICY_DEPTH = 1

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            # Define input placeholders
            obs, legal_actions = split_input(self.processed_obs, ACTIONS)

            # Extract features using a ResNet-like architecture
            extracted_features = resnet_extractor(obs, **kwargs)

            # Policy head
            self._policy = policy_head(extracted_features, legal_actions)
            self._proba_distribution = CategoricalProbabilityDistribution(self._policy)

            # Value head
            self._value_fn = value_head(extracted_features)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                               {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


def split_input(obs, split):
    return obs[:, :, :, :-split], obs[:, :, :, -split:]


def value_head(y):
    for _ in range(VALUE_DEPTH):
        y = dense(y, FEATURE_SIZE)
    vf = dense(y, 1, batch_norm=False, activation='tanh', name='vf')
    return vf


def policy_head(y, legal_actions):
    for _ in range(POLICY_DEPTH):
        y = dense(y, FEATURE_SIZE)
    policy = dense(y, ACTIONS, batch_norm=False, activation=None, name='pi')

    # Apply mask to policy logits
    mask = Lambda(lambda x: (1 - x) * -1e8)(legal_actions)
    policy = Add()([policy, mask])

    return policy


def resnet_extractor(y, **kwargs):
    y = dense(y, FEATURE_SIZE)
    for _ in range(DEPTH):
        y = residual(y, FEATURE_SIZE)

    return y


def residual(y, filters):
    shortcut = y

    y = dense(y, filters)
    y = dense(y, filters, activation=None)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)

    return y


def dense(y, filters, batch_norm=False, activation='relu', name=None):
    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name=name)(y)

    if batch_norm:
        if activation:
            y = BatchNormalization(momentum=0.9)(y)
        else:
            y = BatchNormalization(momentum=0.9, name=name)(y)

    if activation:
        y = Activation(activation, name=name)(y)

    return y
