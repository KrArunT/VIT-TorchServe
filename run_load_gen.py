import os
import time
import requests
import pandas as pd

# Folder containing your ImageNet subsamples (subsampled dataset)
dataset_folder = 'images'
url = "http://127.0.0.1:9085/predictions/vit-model"

headers = {
    'Content-Type': 'application/octet-stream'  # Adjust content type as needed
}

# Function to benchmark server
def benchmark_server(dataset_folder):
    files = [f for f in os.listdir(dataset_folder) if f.endswith('.jpg') or f.endswith('.JPEG') or f.endswith('.png')]
    
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

# Run the benchmark
benchmark_results, total_time, samples_per_second = benchmark_server(dataset_folder)

# Convert results to DataFrame
df = pd.DataFrame(benchmark_results)

# Print benchmark results and samples per second
print("\nBenchmark Results:")
print(df)
print(f"\nTotal Time Taken: {total_time:.2f} seconds")
print(f"Samples Per Second: {samples_per_second:.2f} samples/sec")

# Optionally, save the results to a CSV file
df.to_csv('benchmark_results.csv', index=False)
print("\nBenchmark results saved to 'benchmark_results.csv'")
