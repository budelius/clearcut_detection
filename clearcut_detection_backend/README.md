
```bash

docker build -f postgis.Dockerfile -t clearcut_detection/postgis .
docker build -f django.Dockerfile -t clearcut_detection/backend .
docker build -f model2/model.Dockerfile -t clearcut_detection/model2 .

docker-compose -f docker-compose-wsl.yml up -d db_stage
docker-compose -f ./docker-compose-wsl.yml run --rm django_stage python /code/manage.py migrate --noinput
docker-compose -f ./docker-compose-wsl.yml run --rm django_stage python /code/manage.py loaddata db.json
docker-compose -f ./docker-compose-wsl.yml run --rm django_stage python /code/manage.py createsuperuser

#docker-compose -f docker-compose-wsl.yml up -d

docker-compose -f docker-compose-wsl.yml up

docker-compose -f ./docker-compose-wsl.yml run django_stage python /code/update_all.py --exit-code-from django_stage --abort-on-container-exit django_stage


```

for the api you go to:

http://localhost:9001/api/swagger



download:

https://sentinel.esa.int/documents/247904/1955685/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml


create a runtime configuration for predict_raster with the following options:
--tile_id=32TPT
"--landcover_path=/Volumes/GoogleDrive/My\ Drive/data/landcover/"
"--image_path_current=/Volumes/GoogleDrive/My\ Drive/data/images/sentinel/2022-04-21/32TPT_2022-04-21_output.tif"
"--image_path_previous=/Volumes/GoogleDrive/My\ Drive/data/images/sentinel/32TPT_2022-05-11_output.tif"
"--model_weights_path=/Volumes/GoogleDrive/My\ Drive/data/models/unet_diff.pth"
--save_path=~/Documents/data/out