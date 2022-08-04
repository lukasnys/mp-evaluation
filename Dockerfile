FROM python:3.6

RUN pip3 install pyspark

CMD [“python3”, “generate_workload_remote.py”]