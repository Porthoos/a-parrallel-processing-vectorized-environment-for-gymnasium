import numpy as np
import multiprocessing as mp
from multiprocessing import connection
import gymnasium as gym
from gymnasium.core import Wrapper
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.spaces import Space
from gymnasium.vector.vector_env import VectorEnv
import gym_STAR


def make_env_single(env_id, idx, render_mode=None, FD=None):
    env = gym.make(env_id, render_mode=render_mode, FD=FD)
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    return env


def worker_process(remote, env, seed):
    while True:

        cmd, data = remote.recv()
        if cmd == 'step':
            obs, r, terminated, truncated, infos = env.step(data)
            remote.send((obs, r, terminated, truncated, infos))

        elif cmd == 'reset':
            s, info = env.reset(**data)
            remote.send((s, info))

        elif cmd == 'close':
            env.close()
            remote.close()
            break

        else:
            raise NotImplementedError


class Worker:

    def __init__(self, s, env):
        self.child, parent = mp.Pipe()
        self.ps = mp.Process(target=worker_process, args=(parent, env, s))
        self.ps.start()


class PipeVectorEnv(VectorEnv):

    def __init__(
            self,
            env_id, num_envs=1,
            observation_space: Space = None,
            action_space: Space = None,
            copy: bool = True,
    ):

        self.nb_ps = num_envs
        self.workers = [Worker(i, make_env_single(env_id, idx=i)) for i in range(self.nb_ps)]

        self.copy = copy
        sample_env = gym.make(env_id)
        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or sample_env.observation_space
            action_space = action_space or sample_env.action_space
        super().__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def reset(
            self,
            *,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]

        self._terminateds[:] = False
        self._truncateds[:] = False
        observations, infos = [], {}

        for i, (w, single_seed) in enumerate(zip(self.workers, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options
            w.child.send(('reset', kwargs))

        for i, w in enumerate(self.workers):
            observation, info = w.child.recv()
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def close(self):

        for w in self.workers:
            w.child.send(('close', None))

    def step(self, action):

        for w, a in zip(self.workers, action):
            w.child.send(('step', a))

        observations, infos = [], {}
        for i, w in enumerate(self.workers):
            result = w.child.recv()
            obs, reward, terminated, truncated, info = result
            observations.append(obs)
            self._rewards[i] = reward
            self._terminateds[i] = terminated
            self._truncateds[i] = truncated
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )


if __name__ == '__main__':
    nb_envs = 10
    envs = PipeVectorEnv(env_id="", num_envs=nb_envs)

    print(envs)
    print(envs.reset())

    for _ in range(10):
        # result = envs.step(np.random.randint(0,2, (nb_envs)))
        result = envs.step(np.random.uniform(-1., 1., (nb_envs, 126)))
        print(result[1])

    envs.close()
