# DriftGuard app image: Python + your code + the dashboard
FROM python:3.12-slim

WORKDIR /app

# Install libraries first: Docker caches this layer, so code edits rebuild fast
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the code and install the driftguard package
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/Overview.py", "--server.address=0.0.0.0", "--server.headless=true"]