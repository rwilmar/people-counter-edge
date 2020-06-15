#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = IECore()
        self.input_blob = None
        self.exec_network = None
        ### TODO: Initialize any class variables desired ###

    def load_model(self, model, device, cpu_ext):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        ### TODO: Load the model ###
        self.network = IENetwork(model=model_xml, weights=model_bin)
        ### TODO: Check for supported layers ###
        layers_map = self.plugin.query_network(network=self.network, device_name=device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in layers_map]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Trying to add CPU extensions to IECore.")
        ### TODO: Add any necessary extensions ###
        if cpu_ext and "CPU" in device:
            self.plugin.add_extension(cpu_ext, device)
        ### TODO: Return the loaded inference plugin ###
        self.exec_network = self.plugin.load_network(self.network, device)
        return self.exec_network

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        self.input_blob = next(iter(self.network.inputs))
        return self.network.inputs[self.input_blob].shape
    
    def exec_sync(self, image):
        self.exec_network.infer({self.input_blob: image})
        return self.exec_network.requests[0].outputs

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        request_handler = self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###
        return request_handler
        ### Note: You may need to update the function parameters. ###

    def wait(self, req_handler):
        ### TODO: Wait for the request to be complete. ###
        #status = self.exec_network.requests[request_id].wait(-1)
        status = req_handler.wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self, req_handler):
        ### TODO: Extract and return the output results
        return req_handler.outputs
        ### Note: You may need to update the function parameters. ###
