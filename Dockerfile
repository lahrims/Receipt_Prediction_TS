FROM python:3.11.4
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r requirements.txt
COPY ./models /app/models
COPY ./data /app/data
COPY ./notebooks /app/notebooks
COPY . .
EXPOSE 5000
ENV FLASK_APP=api/app.py
CMD ["flask", "run", "--host", "0.0.0.0"]