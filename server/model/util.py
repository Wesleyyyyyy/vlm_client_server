import os
import configparser
import json
import argparse
from tqdm import tqdm
import logging
import ast
import util as util
import torch


image_extensions = [".jpg", ".jpeg", ".png"]


def get_all_files_in_directory(directory):
    """
    finds all files recursively in a given directory. used to get the input images in a directory

    Args:
        directory (string): directory to find files in recursively.

    Returns:
        list,string: absolute path of all files in the directory.

    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def create_config(config_file):
    """
    Creates the base config file at the location config_file.

    Args:
        config_file (string): Path where to save the config file.

    Returns:
        None.
    """
    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()

    # Default configuration values
    default_config = {
        "General": {
            "model": "llava",
            "input_path": "/-1",
            "output_path": "/-1",
            "prompt": "dummy",
            "prediction_path": "/-1"
        }
    }
    # Write the configuration to the file
    with open(config_file, 'w') as configfile:
        config.write(configfile)


def read_config(args):
    """
    Reads the arguments specified in args and overwrites with those specified in the command line,
    making sure to prioritize the command line arguments over those in the config file.

    Args:
        args (Namespace): Command line arguments (parsed by argparse).

    Returns:
        config: Modified config.
    """
    # Initialize logger for debugging
    logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()
    config.read(args.config)

    # Log the loaded configuration file
    logging.info(f"Loaded config file: {args.config}")

    # Overwrite config values with command line arguments if provided
    for key, value in vars(args).items():
        if value is not None:  # Avoid overwriting with None
            if key in config["General"]:
                logging.info(f"Overwriting '{key}' with value from command line: {value}")
            else:
                logging.warning(f"Key '{key}' not found in config file. Adding it.")
            config["General"][key] = value

    return config


def parse_command_line_args():
    """
    Parses the arguments specified in the command line.
        options:
            --config: path to configuration file
            --model: which model to use, only required for prediction
            --prompt: the prompt to use when predicting the image content, only required for prediction
            --input_path: parent directory of the images, searches images recursively, only required for prediction
            --output_path: path to save prediction at, only required for prediction

    Returns:
        args: the parsed command line arguments.

    """
    parser = argparse.ArgumentParser(description='Command line arguments for prediction')

    # Define all arguments in a more structured way using a dictionary for better readability
    parser.add_argument('--config', help='Specify the path to the configuration file', default='conf/config.ini')
    parser.add_argument('--model', "-m", help='model to use', metavar="model")
    parser.add_argument('--prompt', "-p", help='prompt to use in prediction', metavar="prompt")
    parser.add_argument('--input_path', "-i", help='path of a input image or a directory containing input images', metavar="input_path")
    parser.add_argument('--output_path', "-o", help='path to save predictions to', metavar="output_path")

    args = parser.parse_args()
    return args


def predict(config, model, save_result=True):
    """Runs prediction on images from the specified directory using the given model and save the results

    Args:
        config (dict): Configuration dictionary.
        model: Vision language model.
        save_result (bool): Whether to save predictions.

    Returns:
        None
    """

    input_dir = config["General"].get("input_path", "./data/in/")

    if not os.path.isdir(input_dir):
        print(f"Warning: Input directory '{input_dir}' does not exist.")
        return

    # Check if file extension is correct
    image_paths = [
        img for img in util.get_all_files_in_directory(input_dir)
        if os.path.splitext(img)[-1].lower() in image_extensions
    ]

    if not image_paths:
        print("No valid image files found.")
        return

    print(f"Processing {len(image_paths)} images.")

    prompt = config["General"]["prompt"]
    model_name = config["General"].get("model", "").lower()

    results = {}
    for image_path in tqdm(image_paths):
        # Convert string-formatted list to actual list for CLIP
        if model_name == "clip":
            prompt = ast.literal_eval(prompt)

        result = model.run(image_path, prompt)

        if save_result:
            if "llava" in model_name:
                scores = torch.nn.functional.softmax(result[0]["scores"][0], dim=-1)
                results[image_path] = {
                    "scores": scores.to(torch.float16).cpu().numpy(),
                    "tokens": result[0]["sequences"].to(torch.float16).cpu().numpy(),
                    "text": result[1]
                }
            # Convert string-formatted list to actual list for CLIP
            elif "clip" in model_name:
                results[image_path] = {
                    "prompt": prompt,
                    "scores": result[0].detach().cpu().numpy()}

    if save_result and results:
        results_path = config["General"].get("output_path", "./data/out/")
        os.makedirs(results_path, exist_ok=True)
        results_file = os.path.join(results_path, 'results.json')

        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as file:
                try:
                    results = json.load(file)
                except json.JSONDecodeError:
                    results = []
        else:
            results = []

        results.append(new_data)

        with open(results_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)


def load_prompts(file_path):
    """ Load prompts JSON file and return prompts """
    with open(file_path, "r") as file:
        prompts_data = json.load(file)
    return prompts_data["prompts"]


def process_prompts(prompts, config, model):
    """ Iterate over the prompts and call the prediction function. """
    for prompt in (prompts if isinstance(prompts, list) else [prompts]):
        config["General"]["prompt"] = str(prompt)
        predict(config, model)
