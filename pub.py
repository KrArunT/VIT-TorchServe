import os
import time
import requests
import pandas as pd
import random
import paho.mqtt.client as mqtt
import json

# Folder containing your ImageNet subsamples (subsampled dataset)
dataset_folder = 'images'
url = "http://127.0.0.1:8085/predictions/vit-model"

headers = {
    'Content-Type': 'application/octet-stream'  # Adjust content type as needed
}

# MQTT settings
BROKER = "localhost"
PORT = 1883
TOPIC = "benchmark/metrics"

def get_random_samples(files, sample_size=10):
    # Get all image files
    # Pick random samples from the files list
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
            
            response = requests.post(url, data=f, headers=headers)
            
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
            print(f"Processed {file_name}: Status {response.status_code}, Time Taken: {time_taken:.2f} sec")

    # Calculate samples per second
    if total_time > 0:
        samples_per_second = num_files / total_time
    else:
        samples_per_second = 0

    return results, total_time, samples_per_second

def publish(metrics):
    client = mqtt.Client()

    # Connect to the broker
    client.connect(BROKER, PORT, 60)

    # Publish metrics to the specified topic
    client.publish(TOPIC, json.dumps(metrics))  # Sending the metrics as JSON

    # Wait for the message to be sent
    client.loop()


def send_data():
    # Just run the loop once, so no need for a condition like 'count == 2'
    count = 1
    
    # metrics = {}    
    while True:  # The loop will run once
        # Run the benchmark
        benchmark_results, total_time, samples_per_second = benchmark_server(dataset_folder)

        # Convert results to DataFrame
        df = pd.DataFrame(benchmark_results)

        # Print benchmark results and samples per second
        print("\nBenchmark Results:")
        print(df)
        print(f"\nTotal Time Taken: {total_time:.2f} seconds")
        print(f"Samples Per Second: {samples_per_second:.2f} samples/sec")
        metrics = {} 
        # Prepare metrics for publishing
        metrics = {
            f"sample-{count}": [
                f"Total Time Taken: {total_time:.2f} seconds",
                f"Samples Per Second: {samples_per_second:.2f} samples/sec"
            ]
        }
        
        # Optionally, save the results to a CSV file
        df.to_csv('benchmark_results.csv', index=False)
        df.to_json('benchmark_results.json', index=False, indent=4)
        print("\nBenchmark results saved to 'benchmark_results.csv' and 'benchmark_results.json'")
        
        count += 1
        publish(metrics)
        
    # return metrics


if __name__ == "__main__":
    send_data()
