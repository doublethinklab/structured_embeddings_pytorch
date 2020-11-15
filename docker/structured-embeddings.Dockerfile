FROM timniven/hsdl:base

RUN pip install --upgrade pip
RUN pip install git+https://github.com/timniven/hsdl.git
