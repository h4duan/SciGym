from scigym.agent import Gemini
from scigym.main import setup_controller


def main(
    config_path: str = "configs/default.yml",
    model_name: str = "gemini-2.5-pro-preview-03-25",
):
    controller = setup_controller(config_path, model_name)

    system_prompt = controller._create_system_prompt()
    llm = Gemini(model_name=model_name, system_prompt=system_prompt)

    controller.run_benchmark(model=llm)


if __name__ == "__main__":
    main()
