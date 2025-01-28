# Use pytorch/torchserve:latest as the base image
FROM pytorch/torchserve:latest

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . /app
RUN  pip install --upgrade pip
RUN  pip install -r requirements.txt
# Expose the necessary ports
EXPOSE 9085 9086 9087 9070 9071

# Run the shell script to start the server
CMD ["sh", "run_server.sh"]
