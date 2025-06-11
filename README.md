# 车号(铁路)识别系统

基于YOLO目标检测和OCR文字识别的列车车型车号识别系统，利用DeepSeekR1 优化代码和部分注释，便于理解易读，方便后续学习交流。

## 功能特点

- 支持动车组(CRH)和地铁车型识别
- 基于PaddleOCRv5的高精度车号识别
- 支持YOLO+OCR串联，满足自定义区域识别
- 多线程并行处理

## 系统要求

- Windows 10/11 （理论上支持Linux，未测试）
- Qt 5.14.2
- OpenCV 4.0+
- CUDA 11.8 + cuDNN8.9.7
- OnnxRuntime-gpu 1.16.1

## 目录结构

```
├── algorithm/         # 车号处理算法实现
├── bin/               # 可执行文件及配置
│   ├── Config.ini     # 配置文件
│   ├── Model/         # 模型文件
│   └── Logs/          # 日志文件
├── ocr/               # OCR相关代码
├── src/               # 源代码
├── yolo/              # YOLO检测相关代码
└── 3rdparty/          # 第三方库
```

## 界面

![1](E:\Code\SideTrianNumberRec\SideTrianNumberRec\assert\1.png)

## 参考

OCR-ONNX: https://github.com/Aimol-l/OrtInference

PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR

YOLO-TensorRT:https://github.com/laugh12321/TensorRT-YOLO

Ultralytics: https://github.com/ultralytics/ultralytics



