FROM public.ecr.aws/lambda/python:3.10

# Install TensorFlow 2.15+ or 2.16+ (which includes Keras 3)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY forest_fire_model.h5 ./forest_fire_model.h5
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
