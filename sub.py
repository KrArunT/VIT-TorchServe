import paho.mqtt.client as mqtt
import json

# Define the MQTT Broker settings
broker = "localhost"
port = 1883
topic = "benchmark/metrics"

# Create an MQTT client instance
client = mqtt.Client()

# Define the callback function for message reception
def on_message(client, userdata, message):
    # Convert the JSON message back to a Python dictionary
    data = json.loads(message.payload)
    # Print the dictionary (you can replace this with UI updates)
    print("Received data:", data)

# Connect to the broker
client.connect(broker, port)

# Subscribe to the topic
client.subscribe(topic)

# Attach the message callback function
client.on_message = on_message

# Loop forever, waiting for messages
client.loop_forever()
