# -*- coding: utf-8 -*-
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

"""Execute tune."""

import json
import os
from typing import Any, Dict

from lpot.ux.components.optimization.factory import OptimizationFactory
from lpot.ux.components.optimization.optimization import Optimization
from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.executor import Executor
from lpot.ux.utils.logger import log
from lpot.ux.utils.parser import OptimizationParser
from lpot.ux.utils.templates.workdir import Workdir
from lpot.ux.utils.utils import _load_json_as_dict, get_size
from lpot.ux.web.communication import MessageQueue

mq = MessageQueue()


def execute_optimization(data: Dict[str, Any]) -> dict:
    """Get configuration."""
    from lpot.ux.utils.workload.workload import Workload

    if not str(data.get("id", "")):
        message = "Missing request id."
        mq.post_error(
            "optimization_finish",
            {"message": message, "code": 404},
        )
        raise Exception(message)

    request_id: str = data["id"]
    workdir = Workdir(request_id=request_id, overwrite=False)
    workload_path: str = workdir.workload_path
    try:
        workload_data = _load_json_as_dict(
            os.path.join(workload_path, "workload.json"),
        )
    except Exception as err:
        mq.post_error(
            "optimization_finish",
            {"message": repr(err), "code": 404, "id": request_id},
        )
        raise err
    workload = Workload(workload_data)
    optimization: Optimization = OptimizationFactory.get_optimization(
        workload,
        workdir.template_path,
    )
    send_data = {
        "message": "started",
        "id": request_id,
        "size_input_model": get_size(optimization.input_graph),
    }
    workdir.clean_logs()
    workdir.update_data(
        request_id=request_id,
        model_path=optimization.input_graph,
        input_precision=optimization.input_precision,
        model_output_path=optimization.output_graph,
        output_precision=optimization.output_precision,
        status="wip",
    )

    executor = Executor(
        workspace_path=workload_path,
        subject="optimization",
        data=send_data,
        log_name="output",
    )

    proc = executor.call(
        optimization.command,
    )
    optimization_time = executor.process_duration
    if optimization_time:
        optimization_time = round(optimization_time, 2)
    log.debug(f"Elapsed time: {optimization_time}")
    logs = [os.path.join(workload_path, "output.txt")]
    parser = OptimizationParser(logs)
    if proc.is_ok:
        response_data = parser.process()

        if isinstance(response_data, dict):
            response_data["id"] = request_id
            response_data["optimization_time"] = optimization_time
            response_data["size_optimized_model"] = get_size(optimization.output_graph)
            response_data["model_output_path"] = optimization.output_graph
            response_data["size_input_model"] = get_size(optimization.input_graph)
            response_data["is_custom_dataloader"] = bool(workdir.template_path)

            workdir.update_data(
                request_id=request_id,
                model_path=optimization.input_graph,
                model_output_path=optimization.output_graph,
                metric=response_data,
                status="success",
                execution_details={"optimization": optimization.serialize()},
                input_precision=optimization.input_precision,
                output_precision=optimization.output_precision,
            )
            response_data["execution_details"] = {"optimization": optimization.serialize()}

        log.debug(f"Parsed data is {json.dumps(response_data)}")
        mq.post_success("optimization_finish", response_data)
        return response_data
    else:
        log.debug("FAIL")
        workdir.update_data(
            request_id=request_id,
            model_path=optimization.input_graph,
            input_precision=optimization.input_precision,
            output_precision=optimization.output_precision,
            status="error",
        )
        mq.post_failure("optimization_finish", {"message": "failed", "id": request_id})
        raise ClientErrorException("Optimization failed during execution.")
