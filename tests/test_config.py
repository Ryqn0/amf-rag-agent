from unittest.mock import patch
import amf_rag_agent.config as config
import pytest

def test_validate_raises_when_key_missing():
    with patch.object(config, "ANTHROPIC_API_KEY", ""):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            config.validate_config()