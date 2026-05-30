from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "Contract Predict API"
    app_version: str = "0.1.0"
    allowed_origins: tuple[str, ...] = ("http://localhost:4200",)


settings = Settings()

