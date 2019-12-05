FROM pytorch/pytorch

RUN pip install -r ../requirements.txt
COPY ./ /app
COPY ./docker/entrypoint.sh /app
WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh" ]
CMD []