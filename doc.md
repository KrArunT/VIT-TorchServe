To measure and calculate the **samples per second** (i.e., how many images are processed per second), we can calculate the total time taken for processing all images and then compute the samples per second by dividing the total number of images by the total time.

Here's the updated version of the script that calculates and displays **samples per second**:

```python
import os
import time
import requests
import pandas as pd

# Folder containing your ImageNet subsamples (subsampled dataset)
dataset_folder = 'path_to_imagenet_subsample_folder'
url = "http://127.0.0.1:8085/predictions/vit-model"

headers = {
    'Content-Type': 'application/octet-stream'  # Adjust content type as needed
}

# Function to benchmark server
def benchmark_server(dataset_folder):
    files = [f for f in os.listdir(dataset_folder) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    
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
```

### Key Updates:
1. **Total Time Calculation**: We accumulate the time taken for each image in the `total_time` variable.
2. **Samples Per Second**: After processing all the images, we calculate the samples per second using the formula:
   \[
   \text{Samples Per Second} = \frac{\text{Total Number of Images}}{\text{Total Time Taken}}
   \]
3. **Display the Results**: The script prints the **total time taken** and **samples per second** at the end of the execution.

### Expected Output:
- For each image, the script prints the status and time taken.
- After processing all images, it prints:
  - **Total time taken** (in seconds).
  - **Samples per second** (i.e., how many images the server processes per second on average).

### Example Output:
```plaintext
Processed image1.jpg: Status 200, Time Taken: 0.12 sec
Processed image2.jpg: Status 200, Time Taken: 0.10 sec
...

Benchmark Results:
                   file  status_code                                       prediction  time_taken
0          image1.jpg             200                {'predicted_class': 'Egyptian cat'}         0.12
1          image2.jpg             200                {'predicted_class': 'Egyptian cat'}         0.10
...

Total Time Taken: 6.45 seconds
Samples Per Second: 15.43 samples/sec

Benchmark results saved to 'benchmark_results.csv'
```

This should now give you both the **total time** and the **samples per second**, helping you evaluate the server's performance. Let me know if you need further adjustments!
