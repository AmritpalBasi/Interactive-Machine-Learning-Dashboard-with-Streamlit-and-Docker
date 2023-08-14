# Use an official Python runtime as the parent image
FROM python:3.8

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 8501 available to the world outside this container (Streamlit's default port)
EXPOSE 8501


# Run main.py using Streamlit when the container launches
CMD ["streamlit", "run", "app/main.py"]
