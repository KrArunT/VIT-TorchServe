import pika
import random
import time
import os
import requests
import pandas as pd
import json

# Dummy RabbitMQ details
RABBITMQ_HOST = '192.168.0.26'
RABBITMQ_USER = 'admin'
RABBITMQ_PASSWORD = 'Infobell1234#'

QUEUE_LIST = [
    '1P_LLM_LLAME', '1P_LLM_DS', '1P_VIT', '1P_FW', '1P_POWER',
    '2P_LLM_LLAME', '2P_LLM_DS', '2P_VIT', '2P_FW', '2P_POWER',
    '4PC_LLM_LLAME', '4PC_LLM_DS', '4PC_VIT', '4PC_FW', '4PC_POWER',
]

# Set credentials and connection parameters
CREDENTIALS = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
CONNECTION_PARAMS = pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=CREDENTIALS)

# Folder containing your ImageNet subsamples (subsampled dataset)
dataset_folder = 'images'
url = "http://127.0.0.1:9085/predictions/vit-model"

headers = {
    'Content-Type': 'application/octet-stream'  # Adjust content type as needed
}

# MQTT settings (commented out if not used)
# BROKER = "localhost"
# PORT = 1883
# TOPIC = "1P_LLM_LLAME"

def get_random_samples(files, sample_size=5):
    # Get all image files and pick random samples from the files list
    random_samples = random.sample(files, min(sample_size, len(files)))  # Avoid exceeding the list size
    return random_samples

# Function to benchmark server
def benchmark_server(dataset_folder):
    files = [f for f in os.listdir(dataset_folder) if f.endswith('.jpg') or f.endswith('.JPEG') or f.endswith('.png')]

    files = get_random_samples(files, sample_size=10)
    
    results = []  # To store results
    total_time = 0  # To store total time taken for all requests
    num_files = len(files)  # Total number of images
    
    for file_name in files:
        file_path = os.path.join(dataset_folder, file_name)
        
        # Open the file and send the request
        with open(file_path, 'rb') as f:
            start_time = time.time()  # Start time for benchmarking
            
            try:
                response = requests.post(url, data=f, headers=headers)
                response.raise_for_status()  # Raise exception for non-2xx responses
            except requests.exceptions.RequestException as e:
                print(f"Error processing {file_name}: {e}")
                continue
            
            # Calculate time taken for this image
            time_taken = time.time() - start_time
            total_time += time_taken  # Add to total time
            
            results.append({
                'file': file_name,
                'status_code': response.status_code,
                'prediction': response.json() if response.status_code == 200 else None,
                'time_taken': time_taken
            })
            
            # Optionally print the results for each image
            # print(f"Processed {file_name}: Status {response.status_code}, Time Taken: {time_taken:.2f} sec")

    # Calculate samples per second
    if total_time > 0:
        samples_per_second = num_files / total_time
    else:
        samples_per_second = 0

    return results, total_time, samples_per_second

def publish(queue, message):
    connection = pika.BlockingConnection(CONNECTION_PARAMS)
    channel = connection.channel()

    channel.basic_publish(exchange='',
                            routing_key=queue,
                            body=message,
                            properties=pika.BasicProperties(
                                delivery_mode=2,  # Make message persistent
                            ))
    print(f"Sent to {queue}: {message}")

    # Close connection
    connection.close()

def send_data():
    count = 1
    
    while True:  # The loop will run indefinitely unless manually stopped
        # Run the benchmark
        
        benchmark_results, total_time, samples_per_second = benchmark_server(dataset_folder)

        # Convert results to DataFrame
        df = pd.DataFrame(benchmark_results)

        metrics = {
            f"sample-{count}": [
                f"Total Time Taken: {total_time:.2f} seconds",
                f"Samples Per Second: {samples_per_second:.2f} samples/sec"
            ]
        }

        # Optionally, save the results to a CSV file
        # df.to_csv('benchmark_results.csv', index=False)
        # df.to_json('benchmark_results.json', index=False, indent=4)

        count += 1
        publish('1P_VIT', json.dumps(metrics))  # Using JSON for better structure in the message

if __name__ == "__main__":
    send_data()
