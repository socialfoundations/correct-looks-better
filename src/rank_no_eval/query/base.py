import abc


class BaseQueryMethod(abc.ABC):
    @abc.abstractmethod
    def get_prompt(self, sample: dict) -> str:
        pass

    @abc.abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def completion_kwargs(self) -> dict:
        """Return parameters for API-based model completion."""
        pass
