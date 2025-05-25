import sys
import os

# Add the project root directory to the system path for all tests
sys.path.append(os.path.dirname(os.path.dirname(__file__)) + "/src")

from model_training import train_model


def test_model_accuracy():
	_, accuracy = train_model()

	assert accuracy > .8
