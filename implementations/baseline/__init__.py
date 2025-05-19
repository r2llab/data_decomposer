from core.factory import ImplementationFactory
from .baseline_implementation import BaselineImplementation

# Register Baseline implementation
ImplementationFactory.register('baseline', BaselineImplementation)
