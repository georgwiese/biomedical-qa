# biomedical-qa

## Docker

To start server with single model on port 5000:

```bash
$ docker run -it -p 127.0.0.1:5000:5000 \
  -v `pwd`/final_model:/model \
  georgwiese/biomedical-qa \
  ./start_server.sh single
```
