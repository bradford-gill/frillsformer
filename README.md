# Frillsformer

Python based implementation of the original transformer from [Attention is All You Need](https://arxiv.org/pdf/1706.03762)

## Data

## Running Tests
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


## Cloud Env 
~~~bash
TODO: make sure this works with cuda 
Consider Docker for portability 
~~~

