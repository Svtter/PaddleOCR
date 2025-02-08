#!/bin/bash

# eval data/ folder with custom metrics
VALID_NAME="valid.list"
TEST_NAME="test.list"

# 使用新的配置文件
python tools/eval.py -c atrain_script/configs/ch_PP-OCRv3_rec_distillation_meter.yml \
                     -o Global.checkpoints=./output/rec_ppocr_v3_distillation_meter/best_model/model \
                     Eval.dataset.data_dir=./train_data \
                     Eval.dataset.label_file_list=["./train_data/rec/${TEST_NAME}"] \
                     Metric.name=DistillationMetric \
                     Metric.base_metric_name=MeterRecMetric \
                     Metric.metric_class=mymetric.meter_metric.MeterRecMetric