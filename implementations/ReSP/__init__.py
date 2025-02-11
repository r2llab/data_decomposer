from core.factory import ImplementationFactory
from .resp_implementation import ReSPImplementation

# Register ReSP implementation
ImplementationFactory.register('resp', ReSPImplementation)
