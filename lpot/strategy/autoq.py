#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from copy import deepcopy
import math, time
from ..utils import logger
from ..utils.utility import Timeout
from .strategy import strategy_registry, TuneStrategy
from .ddpg.ddpg import DDPG
from .ddpg.quant_env import QuantEnv


@strategy_registry
class AutoqTuneStrategy(TuneStrategy):
    """The tuning strategy using autoq search in tuning space.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        conf (Conf):                           The Conf class instance initialized from user yaml
                                               config file.
        q_dataloader (generator):              Data loader for calibration, mandatory for
                                               post-training quantization.
                                               It is iterable and should yield a tuple (input,
                                               label) for calibration dataset containing label,
                                               or yield (input, _) for label-free calibration
                                               dataset. The input could be a object, list, tuple or
                                               dict, depending on user implementation, as well as
                                               it can be taken as model input.
        q_func (function, optional):           Reserved for future use.
        eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                               and should yield a tuple of (input, label).
                                               The input could be a object, list, tuple or dict,
                                               depending on user implementation, as well as it can
                                               be taken as model input. The label should be able
                                               to take as input of supported metrics. If this
                                               parameter is not None, user needs to specify
                                               pre-defined evaluation metrics through configuration
                                               file and should set "eval_func" parameter as None.
                                               Tuner will combine model, eval_dataloader and
                                               pre-defined metrics to run evaluation process.
        eval_func (function, optional):        The evaluation function provided by user.
                                               This function takes model as parameter, and
                                               evaluation dataset and metrics should be
                                               encapsulated in this function implementation and
                                               outputs a higher-is-better accuracy scalar value.

                                               The pseudo code should be something like:

                                               def eval_func(model):
                                                    input, label = dataloader()
                                                    output = model(input)
                                                    accuracy = metric(output, label)
                                                    return accuracy
        dicts (dict, optional):                The dict containing resume information.
                                               Defaults to None.

    """

    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None):
        super().__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts)

        # DDPG init
        self.env = QuantEnv(self.opwise_quant_cfgs)
        self.agent = DDPG(nb_states=self.env.get_nb_state(), nb_actions=1, iter_number=300,
                          hparam_override={'warmup_iter_number': 20})

    def params_to_tune_configs(self, params):
        op_cfgs = {}
        op_cfgs['op'] = {}
        for op, configs in self.opwise_quant_cfgs.items():
            if len(configs) > 1:
                value = int(params[op[0]] * len(configs))  # normalized action
                if value == len(configs):
                    value = len(configs) - 1
                op_cfgs['op'][op] = copy.deepcopy(configs[value])
            elif len(configs) == 1:
                op_cfgs['op'][op] = copy.deepcopy(configs[0])
            else:
                op_cfgs['op'][op] = copy.deepcopy(self.opwise_tune_cfgs[op][0])
        if len(self.calib_iter) > 1:
            value = int(params['calib_iteration'])
            if value == len(self.calib_iter):
                value = len(configs) - 1
            op_cfgs['calib_iteration'] = int(self.calib_iter[value])
        else:
            op_cfgs['calib_iteration'] = int(self.calib_iter[0])
        return op_cfgs

    def next_tune_cfg(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.
        """
        best_reward = -math.inf
        episode, episode_reward = 0, 0.
        obs = self.env.get_obs(0)
        transition_buffer = []  # Transition buffer

        while episode < self.cfg.tuning.exit_policy.max_trials:
            action = self.agent.get_next_action(obs, episode)
            next_obs, reward, done = self.env.step(action)
            transition_buffer.append([reward, deepcopy(obs), deepcopy(next_obs), action, done])

            episode_reward += reward
            obs = next_obs

            if done:
                params = self.env.get_quant_params()
                yield self.params_to_tune_configs(params)
                final_reward = self.last_tune_result[0]  # tune accuracy as the reward
                self.train_ddpg(episode, transition_buffer, obs, final_reward)

                # reset
                self.env.reset()
                obs = self.env.get_obs(0)
                episode_reward = 0.
                transition_buffer = []

                value_loss = self.agent.get_value_loss()
                policy_loss = self.agent.get_policy_loss()
                delta = self.agent.get_delta()

                if final_reward > best_reward:
                    best_reward = final_reward

                episode += 1

    def train_ddpg(self, episode, transition_buffer, last_obs, final_reward):
        for i, (_, s_t, _, a_t, done) in enumerate(transition_buffer):
            self.agent.observe(final_reward, s_t, a_t, done)
            if episode >= self.agent.warmup_iter_number:
                for _ in range(self.agent.n_update):
                    self.agent.update_policy()

        # add last pair to memory per HAQ implementation
        self.agent.memory.append(last_obs, self.agent.get_next_action(last_obs, episode=episode), 0., False)

    def get_acc_target(self, base_acc):
        if self.cfg.tuning.accuracy_criterion.relative:
            return base_acc * (1. - self.cfg.tuning.accuracy_criterion.relative)
        else:
            return base_acc - self.cfg.tuning.accuracy_criterion.absolute

    def traverse(self):
        """The main traverse logic, which could be override by some concrete strategy which needs
           more hooks.
        """
        with Timeout(self.cfg.tuning.exit_policy.timeout) as t:
            # get fp32 model baseline
            if self.baseline is None:
                logger.info('Getting FP32 model baseline...')
                self.baseline = self._evaluate(self.model)
                # record the FP32 baseline
                self._add_tuning_history()
            logger.info('FP32 baseline is: ' +
                        ('[{:.4f}, {:.4f}]'.format(*self.baseline) if self.baseline else 'None'))

            # now initiate the HPO here
            # self.experiment = self.create_exp(acc_target=self.get_acc_target(self.baseline[0]))

            trials_count = 0
            for tune_cfg in self.next_tune_cfg():
                # add tune_cfg here as quantize use tune_cfg
                tune_cfg['advance'] = self.cfg.quantization.advance
                trials_count += 1
                tuning_history = self._find_tuning_history(tune_cfg)
                if tuning_history and trials_count < self.cfg.tuning.exit_policy.max_trials:
                    self.last_tune_result = tuning_history['last_tune_result']
                    self.best_tune_result = tuning_history['best_tune_result']
                    logger.debug('This tuning config was evaluated, skip!')
                    continue

                logger.debug('Dump current tuning configuration:')
                logger.debug(tune_cfg)
                self.last_qmodel = self.adaptor.quantize(
                    tune_cfg, self.model, self.calib_dataloader, self.q_func)
                assert self.last_qmodel
                self.last_tune_result = self._evaluate(self.last_qmodel)

                need_stop = self.stop(t, trials_count)

                # record the tuning history
                saved_tune_cfg = copy.deepcopy(tune_cfg)
                saved_last_tune_result = copy.deepcopy(self.last_tune_result)
                self._add_tuning_history(saved_tune_cfg, saved_last_tune_result)

                if need_stop:
                    break
