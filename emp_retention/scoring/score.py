
import json
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('predict-employee-retention')
    model = joblib.load(model_path)


input_sample = np.array([[0.76, 0.5, 4.0, 136.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
output_sample = np.array([1])


@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    y_hat = model.predict(data)
    return y_hat.tolist()
