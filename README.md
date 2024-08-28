# Multi-modal Large Language Model (LLM)

### Introduction
This project involves creating a multi-modal Large Language Model (LLM) capable of processing and integrating text, images, and audio inputs, and generating text-based outputs. The model leverages state-of-the-art models CLIP, Phi, and Whisper to achieve nuanced understanding and interaction across multiple modalities. The final output interface is designed to function similarly to ChatGPT, providing seamless multi-modal AI interactions.

### Requirement
To run and train the multi-modal LLM, the following libraries and frameworks are required:

   * python 3.8
   * torch
   * transformers (HuggingFace)

### Dataset
The COCO128 dataset is used for training the model to generate nuanced image embeddings. This dataset contains a large collection of labeled images, each with various objects, captions, and context, enabling the model to learn the relationships between visual features and textual descriptions. For audio input, preprocessed audio-text datasets are used to fine-tune the Whisper model for accurate audio-to-text conversion.

### Preprocessing
Preprocessing involves several steps for each modality:

Text: Tokenization and normalization of text data to prepare inputs for the Phi Model.
Image: Finetuning of the CLIP model on the COCO dataset to generate meaningful image embeddings.
Audio: Conversion of audio inputs to text using the Whisper model, followed by tokenization and alignment for integration with the Phi Model.
Each input type is transformed to align with the input requirements of the Phi Model and integrated through custom-made projection layers.

### Training and Evaluation
Image Embeddings: The CLIP model is fine-tuned to generate high-quality embeddings for images. A custom projection layer is trained to map these embeddings to the Phi Model's input space.
Model Optimization: QLoRa (Quantized Low-Rank Approximations) is utilized to efficiently reduce the large weight matrices in the Phi Model, improving computational efficiency without sacrificing accuracy.
Audio Integration: Whisper is used to convert audio inputs to text. A custom projection layer is designed to align these inputs with the Phi Model.
Evaluation Metrics: The model's performance is evaluated using standard metrics like accuracy, F1-score, BLEU score for text, and image-caption alignment scores. Extensive testing ensures robust performance across all modalities.

### Resources
   * [COCO128 Dataset](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)
   * CLIP OpenAI: [CLIP GitHub](https://github.com/openai/CLIP)
   * Whisper OpenAI: [Whisper GitHub](https://github.com/openai/whisper)
   * Phi Model Microsoft: [Phi GitHub](https://github.com/microsoft/Phi-3CookBook) 
   * QLoRa: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

# Project structure
```
├──  DATA
│     └── coco128  - here's the default config file.
│         ├── images/train2017
│         ├── labels/train2017
│         └── README.txt
├──  Model
│     ├── clip
│     ├── phi3
│     └── whisper
├──  config
│     ├── __init__.py
│     └── defaults.py
├──  configs
│     └── train_coco_softmax.yml
├──  data
│     ├── datasets
│           ├── __init__.py
│           └── coco.py
│     ├── transforms
│           ├── __init__.py
│           ├── build.py
│           └── transforms.py
│     ├── __init__.py
│     ├── build.py
│     └── collate_batch.py
├──  modeling
├──  notebooks
├──  solver
│     ├── __init__.py
│     ├── build.py
│     └── lr_scheduler.py
├── tools
│     ├── __init__.py
│     └── train_net.py
├── utils
│     ├── __init__.py
│     └── logger.py
```


# Future Work
Future work includes deploying the final multi-modal LLM on the HuggingFace Spaces App to provide a user-friendly interface for public use.



