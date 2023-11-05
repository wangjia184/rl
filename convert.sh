#!/bin/bash

tensorflowjs_converter \
    --control_flow_v2=True \
    --skip_op_check \
    --strip_debug_ops=True \
    --input_format=tf_saved_model \
    --saved_model_tags=serve \
    ./saved_model \
    ./docs/model