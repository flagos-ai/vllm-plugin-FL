# Copyright (c) 2026 BAAI. All rights reserved.
"""Model-owned input and request state hooks for the legacy runner."""


class ModelRunnerState:
    def __init__(self, runner):
        self.runner = runner

    def remove_requests(self, req_ids):
        pass

    def prepare_inputs(self, num_reqs, num_reqs_padded, input_ids):
        return {}

    def prepare_dummy_inputs(self, num_reqs, num_reqs_padded, num_scheduled_tokens):
        return {}
