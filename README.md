# Frillsformer

Python-based implementation of the original transformer from [Attention is All You Need](https://arxiv.org/pdf/1706.03762).

## Data

The Tiny Shakespeare dataset is used for training. It is automatically downloaded when you run the training script.

## Running Tests

To run tests, use the following command:

~~~bash
pytest <pathtofile>
~~~
i.e. pytest tests/src/layers/test_feed_forward.py


## Dev Environment Setup
### Mac 
Use [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html) for virtual env management. 

~~~bash 
# Create venv 
conda create --name frills_p311 python=3.11 --yes
conda deactivate; conda activate frills_p311

# install requirements 
conda install --file requirements.txt
~~~


## Docker Setup for Training

### Building the Docker Image

To build the Docker image, run the following command from the project root:

~~~bash
bash
docker build -t frillsformer .
~~~

### Running the Docker Container

#### With GPU Support

If you have a GPU and the NVIDIA Container Toolkit installed, you can run the container with GPU support:

~~~bash
bash
docker run --gpus all frillsformer
~~~

#### Without GPU Support

If you do not have a GPU or do not wish to use it, simply run:
~~~bash
bash
docker run frillsformer
~~~