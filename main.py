import sys
from loguru import logger

from src import main


if __name__ == "__main__":
    result = main()

    if result is None:
        logger.error("Script execution failed")
        sys.exit(1)
    elif isinstance(result, dict) and "existing_data" in result:
        logger.info("Script completed successfully using existing data")
        sys.exit(0)
    elif isinstance(result, dict) and not result:
        logger.warning("Script completed but no data was retrieved")
        sys.exit(0)
    else:
        logger.info("Script completed successfully")
        sys.exit(0)
