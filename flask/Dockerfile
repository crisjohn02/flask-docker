# start by pulling the python image
FROM python:3.9

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
# COPY ./app /app

COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN ln -s /usr/local/bin/entrypoint.sh /

EXPOSE 5000

# configure the container to run in an executed manner
ENTRYPOINT [ "entrypoint.sh" ]

# CMD ["view.py" ]

CMD [ "entrypoint" ]