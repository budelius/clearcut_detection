
```bash

docker-compose -f docker-compose-dev.yml up -d

docker build -f model.Dockerfile -t clearcut_detection/model2 .

```

for the api you go to:

http://localhost:9001/api/swagger

create a runtime configuration for predict_raster with the following options:
--tile_id=32TPT
"--landcover_path=/Volumes/GoogleDrive/My Drive/data/landcover/"
"--image_path_current=/Volumes/GoogleDrive/My Drive/data/images/sentinel/2022-04-21/32TPT_2022-04-21_output.tif"
"--image_path_previous=/Volumes/GoogleDrive/My Drive/data/images/sentinel/2022-05-11/32TPT_2022-05-11_output.tif"
"--model_weights_path=/Volumes/GoogleDrive/My Drive/data/models/unet_diff.pth"
--save_path=~/Documents/data/out