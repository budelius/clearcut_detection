
```bash

docker-compose -f docker-compose-dev.yml up -d

docker build -f model.Dockerfile -t clearcut_detection/model2 .

```

for the api you go to:

http://localhost:9001/api/swagger