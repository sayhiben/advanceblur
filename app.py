import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Mapping, Sequence, Union

import gradio as gr
import spaces
import torch
from huggingface_hub import hf_hub_download

from nodes import NODE_CLASS_MAPPINGS

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
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


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


@spaces.GPU(duration=360)
def advance_blur(input_image):
    import_custom_nodes()
    with torch.inference_mode():
        load_images_node = NODE_CLASS_MAPPINGS["LoadImagesFromFolderKJ"]()
        source_images_batch = load_images_node.load_images(
            folder="source_faces/",
            width=1024,
            height=1024,
            keep_aspect_ratio="crop",
            image_load_cap=0,
            start_index=0,
            include_subfolders=False,
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loaded_input_image = loadimage.load_image(
            image=input_image,
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscale_model = upscalemodelloader.load_model(
            model_name="ESRGAN/4x_NMKD-Siax_200k.pth"
        )

        reactorbuildfacemodel = NODE_CLASS_MAPPINGS["ReActorBuildFaceModel"]()
        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        reactorfaceswap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            face_model = reactorbuildfacemodel.blend_faces(
                save_mode=True,
                send_only=False,
                face_model_name="default",
                compute_method="Mean",
                images=get_value_at_index(source_images_batch, 0),
            )

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
                face_model=get_value_at_index(face_model, 0),
            )

            upscaled_image = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscale_model, 0),
                image=get_value_at_index(swapped_image, 0),
            )

            final_image = imageresize.execute(
                width=2560,
                height=2560,
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
        gr.Markdown("# Advance Blur")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="filepath")
                generate_btn = gr.Button("Generate")

            with gr.Column():
                # The output image
                output_image = gr.Image(label="Generated Image")

            # When clicking the button, it will trigger the `generate_image` function, with the respective inputs
            # and the output an image
            generate_btn.click(
                fn=advance_blur, inputs=[input_image], outputs=[output_image]
            )
        app.launch(share=True)
