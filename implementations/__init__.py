from core.factory import ImplementationFactory
from .symphony.symphony_implementation import SymphonyImplementation
from .ReSP.resp_implementation import ReSPImplementation
from .XMODE.xmode_implementation import XMODEImplementation
from .baseline.baseline_implementation import BaselineImplementation
from .RAG.rag_implementation import RAGImplementation
# Register implementations
ImplementationFactory.register('symphony', SymphonyImplementation)
ImplementationFactory.register('resp', ReSPImplementation)
ImplementationFactory.register('xmode', XMODEImplementation)
ImplementationFactory.register('rag', RAGImplementation)
ImplementationFactory.register('baseline', BaselineImplementation)