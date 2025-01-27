curl -X POST http://127.0.0.1:8085/predictions/vit-model -T test.jpg & 
curl -X POST http://127.0.0.1:8085/predictions/vit-model -T test.jpg &

#For modifying batch-size
# curl -X POST "localhost:8086/models?model_name=vit-model&url=vit-model.mar&batch_size=2&max_batch_delay=500&initial_workers=3&synchronous=true"
