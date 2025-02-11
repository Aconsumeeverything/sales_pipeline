    # Use an official Python image as the base image
    FROM python:3.9-slim

    # Set the working directory in the container
    WORKDIR /app

    # Copy the local script and the CSV files into the container
    COPY clean.py /app/
    COPY train.csv /app/
    COPY test.csv /app/

    # Install the required Python packages
    RUN pip install --no-cache-dir pandas scikit-learn

    # Run the data cleaning script
    CMD ["python", "clean.py"]
