class BeamLinkError(Exception):
    pass


class ConfigError(BeamLinkError):
    pass


class ArtifactError(BeamLinkError):
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class ShapeError(BeamLinkError):
    def __init__(self, name: str, expected: tuple[int, ...], actual: tuple[int, ...]):
        self.name = name
        self.expected = expected
        self.actual = actual
        super().__init__(f"{name}: expected shape {expected}, got {actual}")
