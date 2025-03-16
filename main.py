import os
import sys
from LLaVA.LLaVA import LLaVA, internlm, deepseek, cog, blip
from pathlib import Path
from util import create_config, read_config, parse_command_line_args, load_prompts, process_prompts, predict
import torch

# Set CUDA visible devices and check for availability
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.getcwd())
print(torch.cuda.is_available())

sys.path.append('./LLaVA/')

models = {
    "llava-1.6_7m": LLaVA,
    "llava-1.6_7v": LLaVA,
    "llava-1.5": LLaVA,
    "llava-1.6_13": LLaVA,
    "llava-1.6_34": LLaVA,
}


def main():
    args = parse_command_line_args()

    config_path = Path(args.config)
    if not config_path.exists():
        create_config(str(config_path))

    config = read_config(args)
    model = models[config["General"]["model"]](config)

    prompts = load_prompts("prompts.json", config["General"]["model"])
    process_prompts(prompts, config, model)


if __name__ == '__main__':
    main()
