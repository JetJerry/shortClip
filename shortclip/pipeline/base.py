import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch


class BaseProcessor(ABC):
    """Base abstract class for all pipeline processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor with configuration.
        
        Args:
            config: Configuration dictionary containing model and processing parameters
        """
        self.config = config
        self.device = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup device from config
        try:
            device_name = config.get("processing", {}).get("device", "cpu")
            # Check for CUDA availability; ensure torch can see GPU
            cuda_available = torch.cuda.is_available()
            cuda_device_count = torch.cuda.device_count() if cuda_available else 0
            
            if device_name == "cuda" and cuda_available and cuda_device_count > 0:
                self.device = torch.device("cuda")
                self.logger.info(f"Using device: {self.device} (GPU: {torch.cuda.get_device_name(0)})")
            else:
                self.device = torch.device("cpu")
                if device_name == "cuda":
                    reason = "not available" if not cuda_available else f"no devices found (count: {cuda_device_count})"
                    self.logger.warning(f"CUDA requested but {reason}, using CPU")
                else:
                    self.logger.info(f"Using device: {self.device}")
        except Exception as e:
            self.device = torch.device("cpu")
            self.logger.error(f"Error setting up device: {e}, defaulting to CPU")
    
    @abstractmethod
    def setup(self) -> None:
        """
        Setup the processor (e.g., load models, initialize resources).
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data.
        
        Args:
            input_data: Input data to process (type depends on processor)
            
        Returns:
            Processed output (type depends on processor)
            
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """
        Cleanup resources (e.g., unload models, close connections).
        Must be implemented by subclasses.
        """
        pass
    
    def _handle_error(self, error: Exception, context: str = "") -> None:
        """
        Simple error handling helper.
        
        Args:
            error: Exception that occurred
            context: Additional context string for the error
        """
        error_msg = f"Error in {self.__class__.__name__}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"
        self.logger.error(error_msg, exc_info=True)
