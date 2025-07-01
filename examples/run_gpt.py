from scigym.agent import GPT
from scigym.main import setup_controller


def main(
    config_path: str = "configs/default.yml",
    model_name: str = "gpt-3.5-turbo",
):
    controller = setup_controller(config_path, model_name)

    system_prompt = controller._create_system_prompt()
    llm = GPT(model_name=model_name, system_prompt=system_prompt)

    controller.run_benchmark(model=llm)


if __name__ == "__main__":
    main()
