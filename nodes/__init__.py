"""
ComfyUI node implementations
"""

# This file makes the nodes directory a proper Python package
# The actual node class imports happen in the main __init__.py

# Import all node classes to make them available
try:
    from .adaptive_enhancement_node import AdaptiveImageEnhancementNode
except ImportError:
    AdaptiveImageEnhancementNode = None

try:
    from .batch_processing_node import BatchImageProcessingNode
except ImportError:
    BatchImageProcessingNode = None

try:
    from .quality_assessment_node import ImageQualityAssessmentNode
except ImportError:
    ImageQualityAssessmentNode = None

try:
    from .film_grain_node import FilmGrainProcessingNode, FilmGrainAnalysisNode
except ImportError:
    FilmGrainProcessingNode = None
    FilmGrainAnalysisNode = None

try:
    from .scunet_node import SCUNetRestorationNode, SCUNetBatchRestorationNode
except ImportError:
    SCUNetRestorationNode = None
    SCUNetBatchRestorationNode = None

# New film grain denoising nodes
try:
    from .fga_nn_film_grain_node import FGANNFilmGrainDenoiseNode
except ImportError:
    FGANNFilmGrainDenoiseNode = None

try:
    from .lightweight_cnn_denoise_node import LightweightCNNDenoiseNode
except ImportError:
    LightweightCNNDenoiseNode = None

# Pre-trained denoising nodes
try:
    from .dncnn_node import DnCNNDenoiseNode
except ImportError:
    DnCNNDenoiseNode = None

try:
    from .nafnet_node import NAFNetDenoiseNode
except ImportError:
    NAFNetDenoiseNode = None

try:
    from .restormer_node import RestormerRestorationNode
except ImportError:
    RestormerRestorationNode = None

__all__ = [
    'AdaptiveImageEnhancementNode',
    'BatchImageProcessingNode', 
    'ImageQualityAssessmentNode',
    'FilmGrainProcessingNode',
    'FilmGrainAnalysisNode',
    'SCUNetRestorationNode',
    'SCUNetBatchRestorationNode',
    'FGANNFilmGrainDenoiseNode',
    'LightweightCNNDenoiseNode',
    'DnCNNDenoiseNode',
    'NAFNetDenoiseNode',
    'RestormerRestorationNode'
]
