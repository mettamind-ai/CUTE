# wget -O hnet/2stage_L.pt https://huggingface.co/cartesia-ai/hnet_2stage_L/resolve/main/hnet_2stage_L.pt

python3 generate_hnet.py --model-path hnet/2stage_L.pt --config-path hnet/2stage_L.json
