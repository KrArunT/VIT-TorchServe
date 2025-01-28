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
## SNORT-ML Deploy
https://github.com/KrArunT/snort-ml-deploy
```sh
docker run --name snort3 -h snort3 -u snorty -w /home/snorty -d -it ciscotalos/snort3 bash  
docker exec -it snort3 bash

docker run --name snort3 -h snort3 -u snorty -w /home/snorty -d -it -v /local_user/SNORT/ml:/home/snorty/examples ciscotalos/snort3 bash

https://archive.wrccdc.org/pcaps/2012/wrccdc2012.pcap.gz  
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

## References:
https://huggingface.co/google/vit-base-patch16-224
