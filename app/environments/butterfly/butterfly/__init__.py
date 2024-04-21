from gym.envs.registration import register

register(
    id='Calico-v0',
    entry_point='calico.envs:CalicoEnvs',
)

