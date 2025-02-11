from core.factory import ImplementationFactory
from .symphony_implementation import SymphonyImplementation

# Register Symphony implementation
ImplementationFactory.register('symphony', SymphonyImplementation) 