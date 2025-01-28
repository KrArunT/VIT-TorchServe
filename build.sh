# Build the Docker image
#docker build -t vit_torchserve_image:v1 .

# Run the Docker container with the required ports
docker run --rm -it -d -p 127.0.0.1:9085:9085 -p 127.0.0.1:9086:9086 -p 127.0.0.1:9087:9087 -p 127.0.0.1:9070:9070 -p 127.0.0.1:9071:9071 vit_torchserve_image:v1
