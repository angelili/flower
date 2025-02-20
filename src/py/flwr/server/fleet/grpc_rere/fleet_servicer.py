# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fleet API gRPC request-response servicer."""


from logging import INFO

import grpc

from flwr.common.logger import log
from flwr.proto import fleet_pb2_grpc
from flwr.proto.fleet_pb2 import (
    PullTaskInsRequest,
    PullTaskInsResponse,
    PushTaskResRequest,
    PushTaskResResponse,
)
from flwr.server.fleet.message_handler import message_handler
from flwr.server.state import State


class FleetServicer(fleet_pb2_grpc.FleetServicer):
    """Fleet API servicer."""

    def __init__(self, state: State) -> None:
        self.state = state

    def PullTaskIns(
        self, request: PullTaskInsRequest, context: grpc.ServicerContext
    ) -> PullTaskInsResponse:
        """Pull TaskIns."""
        log(INFO, "FleetServicer.PullTaskIns")
        return message_handler.pull_task_ins(
            request=request,
            state=self.state,
        )

    def PushTaskRes(
        self, request: PushTaskResRequest, context: grpc.ServicerContext
    ) -> PushTaskResResponse:
        """Push TaskRes."""
        log(INFO, "FleetServicer.PushTaskRes")
        return message_handler.push_task_res(
            request=request,
            state=self.state,
        )
