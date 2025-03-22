!/bin/bash

COMFYUI_DIR=/workspace/ComfyUI
MODELS_DIR=/workspace/models

# TODO: Download Flux 1 D

# TODO: Download Loras

# TODO: Download & install custom nodes
# https://github.com/ltdrdata/ComfyUI-Impact-Subpack
# https://github.com/ltdrdata/ComfyUI-Impact-Pack

# Install ReActor
if [ ! -d $COMFYUI_DIR/custom_nodes/ComfyUI-ReActor ]; then
  cd $COMFYUI_DIR/custom_nodes
  git clone https://github.com/Gourieff/ComfyUI-ReActor
  cd ComfyUI-ReActor
  python install.py
fi

# Download facerestore_models
function hf-pull {
    dest=$1
    url=$2
    filename=$3
    force=$4

    # Create destination directory if it does not exist
    if [ ! -d "$dest" ]; then
        mkdir -p $dest
    fi

    # Skip if destination file already exists and force is not set
    if [ -f "$dest/$filename" ] && [ -z "$force" ]; then
        echo "File $dest/$filename already exists. Skipping."
        return
    fi

    # Content-disposition sets the filename, but we allow overriding it
    if [ -z "$filename" ]; then
        filename=$(basename $url)
    fi
    wget -c $url -O $dest/$filename
}

hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.onnx?download=true"
hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth?download=true"
hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx?download=true" 
hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth?download=true"
hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx?download=true"
hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-2048.onnx?download=true"
hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx?download=true"
hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/RestoreFormer_PP.onnx?download=true"
hf-pull $MODELS_DIR/facerestore_models "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth?download=true"
