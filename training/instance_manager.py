from make_instances import make_n_instances, delete_instances
from screenshot import capture_window


class InstanceManager:
    def __init__(self, num_instances: int) -> None:
        self.num_instances = num_instances
        self.window_titles = [f"AI: {i+1}" for i in range(num_instances)]

    def start_instances(self):
        make_n_instances(self.num_instances)

    def take_picture(name: str):
        return capture_window(name)

    def stop_instances(self):
        delete_instances()


def main() -> None:
    instance = InstanceManager(1)
    instance.stop_instances()


if __name__ == "__main__":
    main()
