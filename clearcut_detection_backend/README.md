
```bash

docker-compose -f docker-compose-dev.yml up -d

docker build -f model.Dockerfile -t clearcut_detection/model2 .

```

for the api you go to:

http://localhost:9001/api/swagger


download:

https://sentinel.esa.int/documents/247904/1955685/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml

