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

import os
import sys
import numpy as np
import subprocess
import signal
import psutil
from ..adaptor import FRAMEWORKS
from ..objective import OBJECTIVES
from ..conf.config import Conf
from ..conf.dotdict import DotDict
from ..utils import logger
from ..utils.create_obj_from_config import create_eval_func, create_dataloader
from ..conf.dotdict import deep_get, deep_set
from ..model import MODELS
from ..model import BaseModel
from .data import TRANSFORMS
from .metric import METRICS
from .common import Model as LpotModel
from .common import Metric as LpotMetric
from .common import Postprocess as LpotPostprocess
from .common import _generate_common_dataloader

def set_env_var(env_var, value, overwrite_existing=False):
    """Sets the specified environment variable. Only set new env in two cases:
        1. env not exists
        2. env already exists but overwirte_existing params set True
    """
    if overwrite_existing or not os.environ.get(env_var):
        os.environ[env_var] = str(value)

def set_all_env_var(conf, overwrite_existing=False):
    # lpot only use physical cores
    cpu_counts = psutil.cpu_count(logical=False)
    if not conf:
        conf = {}
        conf['num_of_instance'] = 1
        conf['cores_per_instance'] = cpu_counts
    if 'cores_per_instance' in conf:
        assert conf['cores_per_instance'] * conf['num_of_instance'] <= cpu_counts,\
            'num_of_instance * cores_per_instance should <= cpu physical cores'
    else:
        assert conf['num_of_instance'] <= cpu_counts, 'num_of_instance should <= cpu counts'
        conf['cores_per_instance'] = int(cpu_counts / conf['num_of_instance'])
    
    for var, value in conf.items():
        set_env_var(var.upper(), value, overwrite_existing)
    # a special but usually used case, directly use current process
    if conf['num_of_instance'] == 1 and conf['cores_per_instance'] == cpu_counts:
        set_env_var('LPOT_ENV_CONF', True, overwrite_existing=True)

class Benchmark(object):
    """Benchmark class can be used to evaluate the model performance, with the objective
       setting, user can get the data of what they configured in yaml
       NOTICE: lpot Benchmark will use the original command to run sub process, this will
       depend on user's code and has possibility to run unneccessary code

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        tuning objective and preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)
        self.framework = self.conf.usr_cfg.model.framework.lower()
        self._model = None
        self._b_dataloader = None
        self._results = {}

    def __call__(self, mode='performance'):
        cfg = self.conf.usr_cfg
        assert cfg.evaluation is not None, 'benchmark evaluation filed should not be None...'
        # use first eval config in yaml if mode from __call__not same with yaml config
        if not mode in cfg.evaluation:
            mode = list(cfg.evaluation.keys())[0]
        assert sys.platform in ['linux', 'win32'], 'only support platform windows and linux...'
        set_all_env_var(deep_get(cfg, 'evaluation.{}.configs'.format(mode)))

        logger.info('Benchmark starts to run...')
        if os.environ.get('LPOT_ENV_CONF') == 'True':
            return self.run_instance(mode)
        else:
            return self.config_instance()

    def config_instance(self):
        raw_cmd = sys.executable + ' ' + ' '.join(sys.argv)
        multi_instance_cmd = ''
        num_of_instance = int(os.environ.get('NUM_OF_INSTANCE'))
        cores_per_instance = int(os.environ.get('CORES_PER_INSTANCE'))
        for i in range(0, num_of_instance):
            core_list = np.arange(0, cores_per_instance) + i * cores_per_instance
            # bind cores only allowed in linux/mac os with numactl enabled
            prefix = self.generate_prefix(core_list)
            instance_cmd = '{} {}'.format(prefix, raw_cmd)
            if sys.platform in ['linux']:
                instance_log = '{}_{}_{}.log'.format(num_of_instance, cores_per_instance, i)
                multi_instance_cmd += '{} 2>&1|tee {} & \\\n'.format(
                    instance_cmd, instance_log)
            else:
                # (TODO) should also add log to win32 benchmark
                multi_instance_cmd += '{} \n'.format(instance_cmd)
        
        multi_instance_cmd += 'wait' if sys.platform in ['linux'] else ''
        logger.info('Instance run command is\n{}'.format(multi_instance_cmd))
        # each instance will execute single instance
        set_env_var('LPOT_ENV_CONF', True, overwrite_existing=True)
        if sys.platform in ['linux']:
            p = subprocess.Popen(multi_instance_cmd, preexec_fn=os.setsid, shell=True) # nosec
        elif sys.platform in ['win32']:
            p = subprocess.Popen(multi_instance_cmd, start_new_session=True, shell=True) # nosec
        try:
            p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

    def generate_prefix(self, core_list):
        if sys.platform in ['linux'] and os.system('numactl --show >/dev/null 2>&1') == 0:
            return 'OMP_NUM_THREADS={} numactl --localalloc --physcpubind={}'.format(\
                len(core_list), ','.join(core_list.astype(str)))
        elif sys.platform in ['win32']:
            # (TODO) should we move the hw_info from ux?
            from lpot.ux.utils.hw_info import get_number_of_sockets
            num_of_socket = int(get_number_of_sockets())
            cores_per_instance = int(os.environ.get('CORES_PER_INSTANCE'))
            cores_per_socket = int(psutil.cpu_count(logical=False)) / num_of_socket
            socket_id = int(core_list[0] // cores_per_socket)
            # cores per socket should integral multiple of cores per instance, else not bind core
            if cores_per_socket % cores_per_instance == 0:
                from functools import reduce
                hex_core = hex(reduce(lambda x, y : x | y, [1 << p for p in core_list]))
                return 'start /b /WAIT /node {} /affinity {} CMD /c'.format(socket_id, hex_core)
        else:
            return ''

    def run_instance(self, mode):
        cfg = self.conf.usr_cfg
        framework_specific_info = {'device': cfg.device, \
                                   'approach': cfg.quantization.approach, \
                                   'random_seed': cfg.tuning.random_seed}
        framework = cfg.model.framework.lower()
        if framework == 'tensorflow':
            framework_specific_info.update({"inputs": cfg.model.inputs, \
                                            "outputs": cfg.model.outputs, \
                                            "recipes": cfg.model.recipes, \
                                            'workspace_path': cfg.tuning.workspace.path})
        if framework == 'mxnet':
            framework_specific_info.update({"b_dataloader": self._b_dataloader})
        if 'onnxrt' in framework.lower():
            framework_specific_info.update({"backend": framework.lower().split('_')[-1], \
                                            'workspace_path': cfg.tuning.workspace.path})
        if framework == 'pytorch_ipex' or framework == 'pytorch':
            framework_specific_info.update({"workspace_path": cfg.tuning.workspace.path,
                                            "q_dataloader": None,
                                            "benchmark": True})

        assert isinstance(self._model, BaseModel), 'need set lpot Model for quantization....'

        adaptor = FRAMEWORKS[framework](framework_specific_info)

        if deep_get(cfg, 'evaluation.{}.iteration'.format(mode)) == -1 and 'dummy_v2' in \
            deep_get(cfg, 'evaluation.{}.dataloader.dataset'.format(mode), {}):
            deep_set(cfg, 'evaluation.{}.iteration'.format(mode), 10)

        iteration = -1 if deep_get(cfg, 'evaluation.{}.iteration'.format(mode)) is None \
            else deep_get(cfg, 'evaluation.{}.iteration'.format(mode))


        metric =  deep_get(cfg, 'evaluation.{}.metric'.format(mode))
        b_postprocess_cfg = deep_get(cfg, 'evaluation.{}.postprocess'.format(mode))

        if self._b_dataloader is None:
            assert deep_get(cfg, 'evaluation.{}.dataloader'.format(mode)) is not None, \
                'dataloader field of yaml file is missing'

            b_dataloader_cfg = deep_get(cfg, 'evaluation.{}.dataloader'.format(mode))
            self._b_dataloader = create_dataloader(self.framework, b_dataloader_cfg)
            b_func = create_eval_func(self.framework, \
                                      self._b_dataloader, \
                                      adaptor, \
                                      metric, \
                                      b_postprocess_cfg,
                                      iteration=iteration)
        else:
            b_func = create_eval_func(self.framework, \
                                      self._b_dataloader, \
                                      adaptor, \
                                      metric, \
                                      b_postprocess_cfg,
                                      iteration=iteration)

        objective = cfg.tuning.objective.lower()
        self.objective = OBJECTIVES[objective](cfg.tuning.accuracy_criterion, \
                                               is_measure=True)

        val = self.objective.evaluate(b_func, self._model)
        logger.info('{} mode benchmark done!'.format(mode))
        # measurer contain info not only performance(eg, memory, model_size)
        # also measurer have result list among steps
        acc, _ = val
        batch_size = self._b_dataloader.batch_size
        warmup =  0 if deep_get(cfg, 'evaluation.{}.warmup'.format(mode)) is None \
            else deep_get(cfg, 'evaluation.{}.warmup'.format(mode))

        assert len(self.objective.measurer.result_list()) > warmup, \
            'itreation should larger than warmup'

        result_list = self.objective.measurer.result_list()[warmup:]
        latency = np.array(result_list).mean() / batch_size
        self._results[mode] = acc, batch_size, result_list

        logger.info('\n{} mode benchmark result:'.format(mode))
        for i, res in enumerate(result_list):
            logger.debug('Iteration {} result {}:'.format(i, res))
        if mode == 'accuracy':
            logger.info('Accuracy is {:.4f}'.format(acc))
        elif mode == 'performance':
            logger.info('Batch size = {}'.format(batch_size))
            logger.info('Latency: {:.3f} ms'.format(latency * 1000))
            logger.info('Throughput: {:.3f} images/sec'.format(1./ latency))

    @property
    def results(self):
        return self._results

    @property
    def b_dataloader(self):
        return self._b_dataloader

    @b_dataloader.setter
    def b_dataloader(self, dataloader):
        """Set Data loader for benchmark, It is iterable and the batched data 
           should consists of a tuple like (input, label) or yield (input, _), 
           when b_dataloader is set, user can configure postprocess(optional) and metric 
           in yaml file or set postprocess and metric cls for evaluation.
           Or just get performance without label in dataloader and configure postprocess/metric.

           Args:
               dataloader(generator): user are supported to set a user defined dataloader
                                      which meet the requirements that can yield tuple of
                                      (input, label)/(input, _) batched data.
                                      Another good practice is to use 
                                      lpot.experimental.common.DataLoader
                                      to initialize a lpot dataloader object.
                                      Notice lpot.experimental.common.DataLoader 
                                      is just a wrapper of the information needed to 
                                      build a dataloader, it can't yield
                                      batched data and only in this setter method 
                                      a 'real' eval_dataloader will be created, 
                                      the reason is we have to know the framework info
                                      and only after the Quantization object created then
                                      framework infomation can be known.
                                      Future we will support creating iterable dataloader 
                                      from lpot.experimental.common.DataLoader

        """
        self._b_dataloader = _generate_common_dataloader(dataloader, self.framework)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, user_model):
        """Set the user model and dispatch to framework specific internal model object

        Args:
           user_model: user are supported to set model from original framework model format
                       (eg, tensorflow frozen_pb or path to a saved model),
                       but not recommended. Best practice is to set from a initialized
                       lpot.experimental.common.Model.
                       If tensorflow model is used, model's inputs/outputs will be 
                       auto inferenced, but sometimes auto inferenced 
                       inputs/outputs will not meet your requests,
                       set them manually in config yaml file.
                       Another corner case is slim model of tensorflow,
                       be careful of the name of model configured in yaml file,
                       make sure the name is in supported slim model list.
        
        """
        if not isinstance(user_model, LpotModel):
            logger.warning('force convert user raw model to lpot model, ' + 
                'better initialize lpot.experimental.common.Model and set....')
            user_model = LpotModel(user_model)

        framework_model_info = {}
        cfg = self.conf.usr_cfg
        if self.framework == 'tensorflow':
            framework_model_info.update(
                {'name': cfg.model.name,
                 'input_tensor_names': cfg.model.inputs,
                 'output_tensor_names': cfg.model.outputs,
                 'workspace_path': cfg.tuning.workspace.path})

        self._model = MODELS[self.framework](\
            user_model.root, framework_model_info, **user_model.kwargs)

    @property
    def metric(self):
        logger.warning('metric not support getter....')
        return None

    @metric.setter
    def metric(self, user_metric):
        """Set metric class and lpot will initialize this class when evaluation
           lpot have many built-in metrics, but user can set specific metric through
           this api. The metric class should take the outputs of the model or 
           postprocess(if have) as inputs, lpot built-in metric always take 
           (predictions, labels) as inputs for update,
           and user_metric.metric_cls should be sub_class of lpot.metric.BaseMetric.

        Args:
            user_metric(lpot.experimental.common.Metric):
                user_metric should be object initialized from
                lpot.experimental.common.Metric, in this method the 
                user_metric.metric_cls will be registered to
                specific frameworks and initialized.
                                              
        """
        assert isinstance(user_metric, LpotMetric), \
            'please initialize a lpot.experimental.common.Metric and set....'

        metric_cfg = {user_metric.name : {**user_metric.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.metric"):
            logger.warning('already set metric in yaml file, will override it...')
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.metric", metric_cfg)
        self.conf.usr_cfg = DotDict(self.conf.usr_cfg)
        metrics = METRICS(self.framework)
        metrics.register(user_metric.name, user_metric.metric_cls)

    @property
    def postprocess(self, user_postprocess):
        logger.warning('postprocess not support getter....')
        return None

    @postprocess.setter
    def postprocess(self, user_postprocess):
        """Set postprocess class and lpot will initialize this class when evaluation. 
           The postprocess class should take the outputs of the model as inputs, and
           output (predictions, labels) as inputs for metric update.
           user_postprocess.postprocess_cls should be sub_class of lpot.data.BaseTransform.

        Args:
            user_postprocess(lpot.experimental.common.Postprocess): 
                user_postprocess should be object initialized from
                lpot.experimental.common.Postprocess,
                in this method the user_postprocess.postprocess_cls will be 
                registered to specific frameworks and initialized.

        """
        assert isinstance(user_postprocess, LpotPostprocess), \
            'please initialize a lpot.experimental.common.Postprocess and set....'
        postprocess_cfg = {user_postprocess.name : {**user_postprocess.kwargs}}
        if deep_get(self.conf.usr_cfg, "evaluation.accuracy.postprocess"):
            logger.warning('already set postprocess in yaml file, will override it...')
        deep_set(self.conf.usr_cfg, "evaluation.accuracy.postprocess.transform", postprocess_cfg)
        postprocesses = TRANSFORMS(self.framework, 'postprocess')
        postprocesses.register(user_postprocess.name, user_postprocess.postprocess_cls)
        logger.info("{} registered to postprocess".format(user_postprocess.name))

