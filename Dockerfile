FROM Python:3.11.5

RUN mkdir /app
WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r requirements.txt
