# TODO: Replace face images with face model to stop creating it every time
# TODO: Figure out .cache error
# TODO: Figure out comfyui-frontend error
# TODO: Preload retinaface_resnet50
# TODO: Preload nsfw model
# TODO: Max out final image size as original image size
# TODO: Double-check inclusion of all necessary custom nodes in repo
# TODO: UI/UX: Better display on mobile so folks don't miss the final output
# TODO: Upgrade gradio

import logging
import os
import sys
from typing import Any, Mapping, Sequence, Union

import gradio as gr
import spaces
import torch
import yaml
from huggingface_hub import hf_hub_download

import folder_paths
from nodes import NODE_CLASS_MAPPINGS

# Load available models from HF
hf_hub_download(
    repo_id="uwg/upscaler",
    filename="ESRGAN/4x_NMKD-Siax_200k.pth",
    local_dir="models/upscale_models",
)
hf_hub_download(
    repo_id="ezioruan/inswapper_128.onnx",
    filename="inswapper_128.onnx",
    local_dir="models/insightface",
)
hf_hub_download(
    repo_id="ziixzz/codeformer-v0.1.0.pth",
    filename="codeformer-v0.1.0.pth",
    local_dir="models/facerestore_models",
)
hf_hub_download(
    repo_id="gmk123/GFPGAN",
    filename="detection_Resnet50_Final.pth",
    local_dir="models/facedetection",
)
hf_hub_download(
    repo_id="gmk123/GFPGAN",
    filename="parsing_parsenet.pth",
    local_dir="models/facedetection",
)
hf_hub_download(
    repo_id="vladmandic/insightface-faceanalysis",
    filename="buffalo_l.zip",
    local_dir="models/insightface/models",
)
hf_hub_download(
    repo_id="model2/advance_face_model",
    filename="advance_face_model.safetensors",
    local_dir="models/reactor/faces",
)


# ReActor has its own special snowflake installation
os.system("cd custom_nodes/ComfyUI-ReActor && python install.py")

def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio

    import execution
    import server
    from nodes import init_extra_nodes

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


# Preload nodes, models.
import_custom_nodes()
loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()
upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
reactorloadfacemodel = NODE_CLASS_MAPPINGS["ReActorLoadFaceModel"]()
FACE_MODEL = reactorloadfacemodel.load_model(
    face_model="advance_face_model.safetensors"
)
imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
reactorfaceswap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
UPSCALE_MODEL = upscalemodelloader.load_model(model_name="ESRGAN/4x_NMKD-Siax_200k.pth")


def load_extra_path_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
            base_path = os.path.expandvars(os.path.expanduser(base_path))
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
        is_default = False
        if "is_default" in conf:
            is_default = conf.pop("is_default")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path:
                    full_path = os.path.join(base_path, full_path)
                elif not os.path.isabs(full_path):
                    full_path = os.path.abspath(os.path.join(yaml_dir, y))
                normalized_path = os.path.normpath(full_path)
                logging.info(
                    "Adding extra search path {} {}".format(x, normalized_path)
                )
                folder_paths.add_model_folder_path(x, normalized_path, is_default)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

@spaces.GPU(duration=60)
def advance_blur(input_image):
    with torch.inference_mode():
        loaded_input_image = loadimage.load_image(
            image=input_image,
        )

        image_size =  getimagesize.execute(
            image=get_value_at_index(loaded_input_image, 0),
        )
        original_width = get_value_at_index(image_size, 0)
        original_height = get_value_at_index(image_size, 1)

        resized_input_image = imageresize.execute(
            width=2560,
            height=2560,
            interpolation="bicubic",
            method="keep proportion",
            condition="downscale if bigger",
            multiple_of=0,
            image=get_value_at_index(loaded_input_image, 0),
        )

        swapped_image = reactorfaceswap.execute(
            enabled=True,
            swap_model="inswapper_128.onnx",
            facedetection="retinaface_resnet50",
            face_restore_model="codeformer-v0.1.0.pth",
            face_restore_visibility=1,
            codeformer_weight=1,
            detect_gender_input="no",
            detect_gender_source="no",
            input_faces_index="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99",
            source_faces_index="0",
            console_log_level=2,
            input_image=get_value_at_index(resized_input_image, 0),
            face_model=get_value_at_index(FACE_MODEL, 0),
        )

        upscaled_image = imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(UPSCALE_MODEL, 0),
            image=get_value_at_index(swapped_image, 0),
        )

        final_image = imageresize.execute(
            width=original_width,
            height=original_height,
            interpolation="lanczos",
            method="keep proportion",
            condition="downscale if bigger",
            multiple_of=0,
            image=get_value_at_index(upscaled_image, 0),
        )

        saved_image = saveimage.save_images(
            filename_prefix="advance_blur",
            images=get_value_at_index(final_image, 0),
        )

        saved_path = f"output/{saved_image['ui']['images'][0]['filename']}"
        return saved_path


if __name__ == "__main__":
    # Start your Gradio app
    with gr.Blocks() as app:
        # Add a title
        gr.Markdown(
            "# Advance Blur"
            "\n\n"
            'Advance Blur uses a sophisticated technique called "Vance Blurring"'
            " to anonymize images of people. This process also removes identifiable"
            " metadata. Uploaded images and data are permanently deleted after processing."
            "\n\n"
            "Advance Blur works best when subjects face the camera. Any similarity to"
            " persons, living or dead, is purely coincidental, comedic, karmic justice,"
            " and/or parody."
            "\n\n"
            "_No sofas, couches, chaises, or other living-room furniture have been harmed in "
            " the production of this application._"
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="filepath")
                generate_btn = gr.Button("Submit")

            with gr.Column():
                # The output image
                output_image = gr.Image(label="Generated Image")

            # When clicking the button, it will trigger the `generate_image` function, with the respective inputs
            # and the output an image
            generate_btn.click(
                fn=advance_blur, inputs=[input_image], outputs=[output_image]
            )
        app.launch(share=True)

        gr.Markdown('#### Have you even said "Thank you"?')
