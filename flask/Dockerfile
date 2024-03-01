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

# # Start by pulling the python image
# FROM python:3.9

# # Copy the requirements file into the image
# COPY ./requirements.txt /app/requirements.txt

# # Switch working directory
# WORKDIR /app

# # Install curl and nodejs
# RUN apt-get update && \
#     apt-get install -y curl && \
#     curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
#     apt-get install -y nodejs

# # Install the dependencies and packages in the requirements file
# RUN pip install -r requirements.txt

# # Copy every content from the local file to the image
# COPY . /app

# # Switch to the Vue.js directory
# WORKDIR /app/frontend

# # Install Vue.js dependencies
# RUN npm install

# # Build the Vue.js application
# RUN npm run build

# # Switch back to the Flask app directory
# WORKDIR /app

# # Expose port 5000
# EXPOSE 5000

# # Make the entrypoint script executable
# COPY entrypoint.sh /usr/local/bin/
# RUN chmod +x /usr/local/bin/entrypoint.sh
# RUN ln -s /usr/local/bin/entrypoint.sh /

# # Configure the container to run in an executed manner
# ENTRYPOINT [ "entrypoint.sh" ]

# # Set the default command to start the Flask app
# CMD [ "python", "view.py" ]