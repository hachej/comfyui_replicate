import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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
    comfyui_path = find_path("interiobot-comfyui")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

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
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import NODE_CLASS_MAPPINGS


class Workflow_preprocessing:
    def __init__(self):
        import_custom_nodes()

    def setup_nodes(self, input_folder):
        self.basefolders3 = NODE_CLASS_MAPPINGS["BaseFolderS3"]()
        self.basefolders3_5 = self.basefolders3.return_base_folder(
            base_folder=input_folder
        )
        self.loadimages3 = NODE_CLASS_MAPPINGS["LoadImageS3"]()
        self.sampreprocessor = NODE_CLASS_MAPPINGS["SAMPreprocessor"]()
        self.saveimages3 = NODE_CLASS_MAPPINGS["SaveImageS3"]()
        self.pidinetpreprocessor = NODE_CLASS_MAPPINGS["PiDiNetPreprocessor"]()

    def run(self, input_folder):
        with torch.inference_mode():
            self.setup_nodes(input_folder)
            self.loadimages3_4 = self.loadimages3.load_image_s3(
                bucket_name="sum-int-main-app-interior-processed",
                base_folder=get_value_at_index(self.basefolders3_5, 0),
                s3_path="image_resized.png",
            )

            self.sampreprocessor_1 = self.sampreprocessor.execute(
                resolution=512, image=get_value_at_index(self.loadimages3_4, 0)
            )

            self.saveimages3_8 = self.saveimages3.save_images_s3(
                bucket_name="sum-int-main-app-interior-processed",
                s3_base_folder=get_value_at_index(self.basefolders3_5, 0),
                s3_path="preprocessing/image_segmented",
                images=get_value_at_index(self.sampreprocessor_1, 0),
            )

            self.pidinetpreprocessor_12 = self.pidinetpreprocessor.execute(
                safe="enable",
                resolution=512,
                image=get_value_at_index(self.loadimages3_4, 0),
            )

            self.saveimages3_13 = self.saveimages3.save_images_s3(
                bucket_name="sum-int-main-app-interior-processed",
                s3_base_folder=get_value_at_index(self.basefolders3_5, 0),
                s3_path="image_segmented_lines",
                images=get_value_at_index(self.pidinetpreprocessor_12, 0),
            )


if __name__ == "__main__":
    w = Workflow_preprocessing()
    w.run("tests/test_image_06/")
