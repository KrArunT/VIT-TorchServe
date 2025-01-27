torchserve --start \
           --ncs \
           --model-store model-store \
           --models vit-model=vit-model.mar \
           --ts-config config.properties \
           --disable-token-auth \
           --enable-model-api

