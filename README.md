# LLama3.2:1B Deployment
## Steps
```sh
docker pull vllm/vllm-openai:latest
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<your_hugging_face_token>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.2-1B-Instruct

```

# VIT Deployment
## **Run the Docker container with the required ports**
```sh
docker run --rm -it -d -p 127.0.0.1:9085:9085 -p 127.0.0.1:9086:9086 -p 127.0.0.1:9087:9087 -p 127.0.0.1:9070:9070 -p 127.0.0.1:9071:9071 aruntiwary/vit_torchserve_image:v1
```

## For running benchmark
* Download Imagenet 9K samples.
  ```sh
  wget https://github.com/EliSchwartz/imagenet-sample-images
  mv imagenet-sample-images images
  ```

### Run Loadgen and benchmark
```sh
python3 -m venv env
source env/bin/activate
pip install pandas request
python run_load_gen.py
```

## Build the Docker image (Optional)
#docker build -t vit_torchserve_image:v1 .

## Create Virtual env
```sh
git clone https://github.com/KrArunT/VIT-TorchServe.git
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
mkdir model-store
./package.sh
./run_server.sh
#Open New terminal and run
# Download image
wget 'http://images.cocodataset.org/val2017/000000039769.jpg'
mv 000000039769.jpg test.jpg
./infer.sh
```



## Save Model
```sh
from transformers import ViTImageProcessor, ViTForImageClassification

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

processor.save_pretrained('./vit_model')
model.save_pretrained('./vit_model')

```

## Config.Properties
```sh
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8086
metrics_address=http://0.0.0.0:8087

# Set the batch size to 8
inference_batch_size=8
max_batch_delay=100
```

## Handler.py
```sh
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from ts.torch_handler.base_handler import BaseHandler
from io import BytesIO

class ImageClassificationHandler(BaseHandler):
    """
    A custom handler for Image Classification using Vision Transformer (ViT).
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None

    def initialize(self, context):
        """
        Initialize the model and processor during model loading.
        :param context: The context provides system properties like model_dir, gpu_id, etc.
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_name = "google/vit-base-patch16-224"  # Model to be loaded from Hugging Face Hub

        # Initialize the processor and model from Hugging Face Hub
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

        # Move model to the appropriate device (GPU or CPU)
        device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.initialized = True


    # #Single image: batch-1
    def preprocess(self, data):
        """
        Preprocess the raw input data before inference.
        :param data: The raw input data, typically image data.
        :return: Preprocessed input ready for model inference.
        """
        # Assuming the data contains an image in the 'body' field.
        # image = Image.open(data[0]['body'])
        # image = Image.open(BytesIO(image_data))

        image_data = data[0]['body']
        # Convert the bytearray to a BytesIO stream, which is file-like
        image = Image.open(BytesIO(image_data))

        # Preprocess the image using the processor
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs


    def inference(self, model_input):
        """
        Run inference using the model.
        :param model_input: The preprocessed data.
        :return: The model's prediction output.
        """
        with torch.no_grad():
            outputs = self.model(**model_input)
        
        logits = outputs.logits
        return logits


    def postprocess(self, inference_output):
        """
        Postprocess the inference output.
        :param inference_output: The raw model output (logits).
        :return: Processed output (predicted class).
        """
        # Get the predicted class index from the logits
        predicted_class_idx = inference_output.argmax(-1).item()

        # Assuming that the model's config contains id2label mapping
        predicted_class = self.model.config.id2label[predicted_class_idx]

        preds = {}
        preds["predicted_class_idx"]=predicted_class_idx
        preds["predicted_class"]=[predicted_class]
        return [preds]
    

    def handle(self, data, context):
        """
        The main handler function invoked by TorchServe for predictions.
        It performs preprocessing, inference, and postprocessing.
        :param data: Input data for prediction.
        :param context: Context containing system properties.
        :return: Prediction output (the predicted class).
        """
        # Preprocess the data
        model_input = self.preprocess(data)

        # Run inference
        inference_output = self.inference(model_input)

        # Postprocess the result
        return self.postprocess(inference_output)

```

## Package Model Archieve (MAR)
```sh
torch-model-archiver --model-name vit-model \
                     --version 2.0 \
                     --handler handler.py \
                     --extra-files ./vit_model \
                     --export-path model-store \
                     --force

```

## Run Server
```sh
torchserve --start \
           --ncs \
           --model-store model-store \
           --models vit-model=vit-model.mar \
           --ts-config config.properties \
           --disable-token-auth \
           --enable-model-api
```

## Test Inferenence
```sh
curl -X POST http://127.0.0.1:8085/predictions/vit-model -T test.jpg & 
curl -X POST http://127.0.0.1:8085/predictions/vit-model -T test.jpg &

#For modifying batch-size
# curl -X POST "localhost:8086/models?model_name=vit-model&url=vit-model.mar&batch_size=2&max_batch_delay=500&initial_workers=3&synchronous=true"

```
* Test Image ![image](https://github.com/user-attachments/assets/9f6570d2-d664-4582-8a21-08d217a420dc)


## Reqs
```sh
certifi==2024.12.14
charset-normalizer==3.4.1
enum-compat==0.0.3
filelock==3.17.0
fsspec==2024.12.0
huggingface-hub==0.27.1
idna==3.10
Jinja2==3.1.5
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.2
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
packaging==24.2
pillow==11.1.0
psutil==6.1.1
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
safetensors==0.5.2
sympy==1.13.1
tokenizers==0.21.0
torch==2.5.1
torch-model-archiver==0.12.0
torchserve==0.12.0
tqdm==4.67.1
transformers==4.48.1
triton==3.1.0
typing_extensions==4.12.2
urllib3==2.3.0

```

## Stop Server
```sh
torchserve --stop

```
## References:
https://huggingface.co/google/vit-base-patch16-224
