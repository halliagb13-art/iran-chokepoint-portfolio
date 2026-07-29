from src.gpr.base import GPRAdapter
from src.gpr.published_adapter import PublishedGPRAdapter
from src.gpr.zen_adapter import ZenGPRAdapter
from src.config import OPENCODE_API_KEY


def create_gpr_adapter(
    method: str = "published",
) -> GPRAdapter:
    if method == "zen":
        return ZenGPRAdapter()
    elif method == "deepseek":
        return ZenGPRAdapter()
    elif method == "published":
        return PublishedGPRAdapter(use_mideast=True)
    elif method == "auto":
        if OPENCODE_API_KEY:
            return ZenGPRAdapter()
        return PublishedGPRAdapter(use_mideast=True)
    else:
        raise ValueError(f"Unknown GPR method: {method}")
