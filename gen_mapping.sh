#!/bin/bash

./target/debug/linguagraph generate-mapping examples/teye/teye_data.json \
--collections cameras,places,events \
--ontology-domain cameras --ontology-file .linguagraph/ontology_catalog.json \
--base-url http://100.79.136.128:8001/v1 --model /root/.cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/e850c696e6d75f965367e816c16bc7dacd955ffa \
--describe