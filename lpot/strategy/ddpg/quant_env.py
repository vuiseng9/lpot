"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import copy
from collections import OrderedDict
import torch.nn as nn
import numpy as np
from ...utils import logger


class QuantEnv:
    def __init__(self, opwise_quant_cfgs):
        self.cur_step = 0
        # embedding node: [cur_state, op, cur_action]
        self.state_embedding, self.max_cfg = self.build_embedding(opwise_quant_cfgs)

    def step(self, action):
        # store current action for current state
        self.state_embedding[self.cur_step][2] = action
        if self.cur_step < len(self.state_embedding) - 1:
            self.state_embedding[self.cur_step + 1][0]['prev_action'] = action
            obs = self.get_obs(self.cur_step + 1)
        else:
            obs = self.get_obs(len(self.state_embedding) - 1)
        done = self.cur_step == len(self.state_embedding) - 1
        self.cur_step += 1
        return obs, 0, done

    def reward(self):
        pass

    def reset(self):
        self.cur_step = 0

    def build_embedding(self, opwise_quant_cfgs):
        embedding, max_cfg = [], 0
        for (name, type, m), configs in opwise_quant_cfgs.items():
            feature = OrderedDict()
            if len(configs) <= 1:
                continue
            feature['layer_idx'] = len(embedding)
            max_cfg = len(configs) if len(configs) > max_cfg else max_cfg
            if isinstance(m, (nn.intrinsic.ConvReLU2d)):
                m = m._modules['0']
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                feature['conv_dw'] = int(m.weight.shape[1] == m.groups) # 1.0 for depthwise, 0.0 for other conv2d
                feature['cin'] = m.weight.shape[1]
                feature['cout'] = m.weight.shape[0]
                feature['stride'] = m.stride[0]
                feature['kernel'] = m.kernel_size[0]
                feature['param'] = np.prod(m.weight.size())
                # kding1: todo
                # feature['ifm_size'] = np.prod(m.input_shape_[-2:]) # H*W
                feature['prev_action'] = 0.0 # placeholder
            elif isinstance(m, nn.Linear):
                feature['conv_dw'] = 0.0
                feature['cin'] = m.in_features
                feature['cout'] = m.out_features
                feature['stride'] = 0.0
                feature['kernel'] = 1.0
                feature['param'] = np.prod(m.weight.size())
                # kding1: todo
                # feature['ifm_size'] = np.prod(m.input_shape_[-1]) # feature nodes
                feature['prev_action'] = 0.0 # placeholder
            else:
                raise NotImplementedError("State embedding extraction of {}".format(m.__class__.__name__))

            # kding1: todo
            # layer_attr_df['weight_quantizer'] = df['is_wt_quantizer'].astype('float')
            embedding.append([feature, name, 0.])

        # normalize embedding
        for k in embedding[0][0].keys():
            v = [e[0][k] for e in embedding]
            fmin, fmax = min(v), max(v)
            for e in embedding:
                e[0][k] = (e[0][k] - fmin) / (fmax - fmin) if fmax - fmin > 0 else 0.

        return embedding, max_cfg

    def get_nb_state(self):
        return len(self.state_embedding[0][0])

    def get_obs(self, idx):
        return list(self.state_embedding[idx][0].values())

    def get_quant_params(self):
        return dict([(e[1], e[2]) for e in self.state_embedding])
