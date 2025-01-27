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

