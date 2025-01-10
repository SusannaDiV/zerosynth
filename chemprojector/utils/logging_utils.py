import logging
import os

def setup_logging(log_dir='logs'):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging configuration
    logging.basicConfig(
        filename=os.path.join(log_dir, 'pharmacophore_debug.log'),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Overwrite any existing logging configuration
    ) 