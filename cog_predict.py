from typing import Any
from cog import BasePredictor, Input, Path
import numpy as np
from workflows import workflow_preprocessing 

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = workflow_preprocessing.Workflow_multiregion() 

    # Define the arguments and types the model takes as input
    def predict(self) -> Any:
        """Run a single prediction on the model"""
        # Preprocess the image
        self.model.run("tests/test_image_06/")
        # Run the prediction
        return {"status": "ohyeah"}

if __name__ == "__main__":
    p = Predictor()
    p.setup()
    p.predict()