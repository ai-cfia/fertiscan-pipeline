import unittest
from pydantic import ValidationError
from pipeline.schemas.settings import Settings
from pydantic import SecretStr

class TestSettings(unittest.TestCase):
    def setUp(self):
        self.valid_data = {
            "AZURE_API_ENDPOINT": "https://fertiscan-document-model.com/",
            "AZURE_API_KEY": "random-key123",
            "AZURE_OPENAI_ENDPOINT": "https://api.openai.com/",
            "AZURE_OPENAI_KEY": "random-key123",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "",
        }

        self.invalid_data = {
            "AZURE_API_ENDPOINT": "invalid-url",
            "AZURE_API_KEY": 12345,  # Should be a string
            "AZURE_OPENAI_ENDPOINT": "invalid-url",
            "AZURE_OPENAI_KEY": 12345,  # Should be a string
            "AZURE_OPENAI_DEPLOYMENT": 12345,  # Should be a string
            "OTEL_EXPORTER_OTLP_ENDPOINT": 12345,  # Should be a string or None
        }

    def test_valid_settings(self):
        settings = Settings(**self.valid_data)
        self.assertEqual(settings.document_api_endpoint, self.valid_data["AZURE_API_ENDPOINT"])
        self.assertEqual(settings.document_api_key, SecretStr(self.valid_data["AZURE_API_KEY"]))
        self.assertEqual(settings.llm_api_endpoint, self.valid_data["AZURE_OPENAI_ENDPOINT"])
        self.assertEqual(settings.llm_api_key, SecretStr(self.valid_data["AZURE_OPENAI_KEY"]))
        self.assertEqual(settings.llm_api_deployment, self.valid_data["AZURE_OPENAI_DEPLOYMENT"])
        self.assertEqual(settings.otel_exporter_otlp_endpoint, self.valid_data["OTEL_EXPORTER_OTLP_ENDPOINT"])

    def test_invalid_settings(self):
        with self.assertRaises(ValidationError):
            Settings(**self.invalid_data)

    def test_env_file(self):
        try:
            Settings()
        except ValidationError:
            self.fail("Settings could not be created from environment file")

if __name__ == "__main__":
    unittest.main()
