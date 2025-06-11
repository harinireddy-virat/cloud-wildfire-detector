import json
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import boto3
import os

# Load model
model = tf.keras.models.load_model("/var/task/forest_fire_model.h5", compile=False)


print(type(model))

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Validate input keys
    if 'image_bucket' not in event or 'image_key' not in event:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Please provide 'image_bucket' and 'image_key' in the request"}),
            "headers": {"Content-Type": "application/json"}
        }

    bucket = event['image_bucket']
    key = event['image_key']

    try:
        # Download image from S3
        s3_response = s3.get_object(Bucket=bucket, Key=key)
        img_bytes = s3_response['Body'].read()

        # Preprocess image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((150,150))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Predict
        score = float(model.predict(arr)[0][0])
        label = "No Fire" if score >= 0.5 else "Fire"


        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction_score": score,
                "predicted_class": label
            }),
            "headers": {"Content-Type": "application/json"}
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
