torch-model-archiver --model-name vit-model \
                     --version 2.0 \
                     --handler handler.py \
                     --extra-files ./vit_model \
                     --export-path model-store \
                     --force
