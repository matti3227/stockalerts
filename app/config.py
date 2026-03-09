from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/stock_alerts"
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "StockAlerts/1.0"
    stocktwits_access_token: Optional[str] = None
    scrape_interval_minutes: int = 15

    model_config = {"env_file": ".env"}


settings = Settings()
