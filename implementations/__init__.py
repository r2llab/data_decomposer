from core.factory import ImplementationFactory
from .symphony.symphony_implementation import SymphonyImplementation
from .ReSP.resp_implementation import ReSPImplementation

# Register implementations
ImplementationFactory.register('symphony', SymphonyImplementation)
ImplementationFactory.register('resp', ReSPImplementation) 