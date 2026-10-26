#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型推理封装

包路径：detection.onnx_infer
"""
import os
import onnxruntime as ort

from detection.traffic_sign_config import MODEL_PATH


class OnnxRun:
    """加载 ONNX 模型并提供推理接口。"""

    def __init__(self, model_name="", model_path=MODEL_PATH,
                 providers=None):
        """
        :param model_name: 模型名称（仅用于日志显示）
        :param model_path: ONNX 模型文件路径
        :param providers: 推理后端，默认优先 CUDA，回退 CPU
        """
        self.model_name = model_name
        self.model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'ONNX 模型不存在: {model_path}')

        if providers is None:
            # 优先 CUDA，仅选用本机实际可用的后端，避免不必要的警告
            available = set(ort.get_available_providers())
            providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                         if p in available]
            if not providers:
                providers = ['CPUExecutionProvider']

        config = ort.SessionOptions()
        config.intra_op_num_threads = 1
        config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(
            model_path, sess_options=config, providers=providers)

        self.input_name = self.ort_session.get_inputs()[0].name

        print(f'[{model_name or "OnnxRun"}] input: {self.ort_session.get_inputs()[0]}')
        for out in self.ort_session.get_outputs():
            print(f'[{model_name or "OnnxRun"}] output: {out}')

    def inference(self, img):
        """执行推理，返回模型输出的 list。"""
        return self.ort_session.run(None, {self.input_name: img})


if __name__ == '__main__':
    run = OnnxRun()
    print('推理后端可用，模型加载成功')