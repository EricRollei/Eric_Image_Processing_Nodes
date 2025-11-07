 # Eric_Image_Processing_Nodes/__init__.py
"""
Eric's Image Processing Nodes for ComfyUI
Advanced image enhancement, denoising, restoration, and frequency domain techniques
GPU-accelerated processing with specialized film grain handling
"""

# Initialize all mapping dictionaries to prevent NameError
WAVELET_MAPPINGS = WAVELET_DISPLAY = {}
NLM_MAPPINGS = NLM_DISPLAY = {}
RL_MAPPINGS = RL_DISPLAY = {}
WIENER_MAPPINGS = WIENER_DISPLAY = {}
FREQ_MAPPINGS = FREQ_DISPLAY = {}
ADAPT_MAPPINGS = ADAPT_DISPLAY = {}
BATCH_MAPPINGS = BATCH_DISPLAY = {}
QUALITY_MAPPINGS = QUALITY_DISPLAY = {}
FILM_GRAIN_MAPPINGS = FILM_GRAIN_DISPLAY = {}
NOISE_DA_MAPPINGS = NOISE_DA_DISPLAY = {}
ADVANCED_MAPPINGS = ADVANCED_DISPLAY = {}
ADVANCED_AI_MAPPINGS = ADVANCED_AI_DISPLAY = {}
SCUNET_MAPPINGS = SCUNET_DISPLAY = {}
NEW_ADVANCED_MAPPINGS = NEW_ADVANCED_DISPLAY = {}
AUTO_DENOISE_MAPPINGS = AUTO_DENOISE_DISPLAY = {}
BM3D_MAPPINGS = BM3D_DISPLAY = {}
ADVANCED_SHARPENING_MAPPINGS = ADVANCED_SHARPENING_DISPLAY = {}
LEARNING_CLAHE_MAPPINGS = LEARNING_CLAHE_DISPLAY = {}
ADAPTIVE_FREQUENCY_MAPPINGS = ADAPTIVE_FREQUENCY_DISPLAY = {}
CUTTING_EDGE_COMPARISON_MAPPINGS = CUTTING_EDGE_COMPARISON_DISPLAY = {}
REAL_BM3D_MAPPINGS = REAL_BM3D_DISPLAY = {}
FGANN_MAPPINGS = FGANN_DISPLAY = {}
PROGRESSIVE_CNN_MAPPINGS = PROGRESSIVE_CNN_DISPLAY = {}
DNCNN_MAPPINGS = DNCNN_DISPLAY = {}
NAFNET_MAPPINGS = NAFNET_DISPLAY = {}
DEEPINV_MAPPINGS = DEEPINV_DISPLAY = {}

# Import all functions from scripts
try:
    # Frequency enhancement functions
    from .scripts.frequency_enhancement import (
        homomorphic_filter,
        phase_preserving_enhancement,
        multiscale_fft_enhancement,
        adaptive_frequency_filter,
        get_frequency_enhancement_presets
    )
    
    # Wavelet denoising functions (including GPU-accelerated)
    from .scripts.wavelet_denoise import (
        wavelet_denoise,
        wavelet_denoise_stationary,
        gpu_wavelet_denoise,
        gpu_wavelet_denoise_stationary,
        estimate_noise_level,
        get_available_wavelets
    )
    
    # Non-local means functions
    from .scripts.nonlocal_means import (
        nonlocal_means_denoise,
        adaptive_nonlocal_means,
        get_recommended_parameters
    )
    
    # Richardson-Lucy functions
    from .scripts.richardson_lucy import (
        richardson_lucy_deconvolution,
        get_blur_presets,
        estimate_motion_blur,
        create_motion_psf,
        create_gaussian_psf
    )

    from .scripts.richardson_lucy_gpu import (
        richardson_lucy_deconvolution_gpu,
    )
    
    # Wiener filter functions
    from .scripts.wiener_filter import (
        wiener_filter_restoration,
        adaptive_wiener_filter,
        parametric_wiener_filter,
        get_wiener_presets
    )
    
    # GPU utilities
    from .scripts.gpu_utils import (
        get_gpu_info,
        gpu_memory_info,
        cleanup_gpu_memory,
        can_use_gpu,
        gpu_gaussian_blur,
        gpu_bilateral_filter,
        gpu_non_local_means,
        gpu_frequency_filter
    )
    
    # Advanced PSF modeling functions
    from .scripts.advanced_psf_modeling import (
        process_with_psf_modeling,
        get_psf_presets,
        AdvancedPSFProcessor
    )

    # Advanced sharpening functions
    from .scripts.advanced_sharpening import (
        AdvancedSharpeningProcessor
    )

    # Perceptual color enhancement functions
    from .scripts.perceptual_color_processing import (            
        process_with_perceptual_color,
        get_perceptual_color_presets,
        PerceptualColorProcessor
    )   

    # Film grain and advanced processing
    from .scripts.film_grain_processing import (
        analyze_grain_type,
        denoise_film_grain,
        get_grain_processing_recommendations
    )
    
    # Advanced film grain processing
    from .scripts.advanced_film_grain import (
        FilmGrainProcessor,
        FilmGrainAnalyzer
    )
    
    # Noise-DA processing
    from .scripts.noise_da_processing import (
        NoiseDAProcessor
    )
    
    # Advanced traditional processing
    from .scripts.advanced_traditional_processing import (
        LBCLAHEProcessor,
        MultiScaleRetinexProcessor,
        BM3DGTADProcessor,
        SmartSharpeningProcessor
    )
    
    # Learning-based CLAHE
    from .scripts.learning_based_clahe import (
        LearningBasedCLAHEProcessor
    )
    
    # Adaptive frequency decomposition
    from .scripts.adaptive_frequency_decomposition import (
        AdaptiveFrequencyDecompositionProcessor
    )
    
    # Auto denoise functions
    from .scripts.auto_denoise import (
        Noise2VoidProcessor,
        DeepImagePriorProcessor,
        AutoDenoiseProcessor
    )
    
    # BM3D denoising
    from .scripts.bm3d_denoise import (
        BM3DProcessor
    )
    
    # AI Processing scripts
    from .scripts.real_esrgan_processing import (
        RealESRGANProcessor,
        get_realesrgan_presets
    )
    
    from .scripts.sfhformer_processing import (
        SFHformerProcessor,
        get_sfhformer_presets
    )
    
    # SCUNet processing
    from .scripts.scunet_processing import (
        SCUNetProcessor
    )
    
    from .scripts.practical_scunet import (
        PracticalSCUNetProcessor
    )
    
    from .scripts.simplified_scunet import (
        SimplifiedSCUNetProcessor
    )
    
    # SwinIR processing
    from .scripts.swinir_processing import (
        SwinIRProcessor
    )

    # Restormer processing
    from .scripts.restormer_processing import (
        RestormerProcessor,
        MODEL_CONFIGS as RESTORMER_MODEL_CONFIGS
    )

    # DiffBIR processing
    from .scripts.diffbir_processing import (
        DiffBIRConfig,
        DiffBIRProcessor
    )
    
    # Memory utilities
    from .scripts.memory_utils import (
        MemoryManager
    )
    
    # Base processing node
    from .base_node import BaseImageProcessingNode
    
    # Import all node classes
    try:
        from .nodes.wavelet_denoise_node import WaveletDenoiseNode
        WAVELET_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Wavelet denoise node not available: {e}")
        WaveletDenoiseNode = None
        WAVELET_AVAILABLE = False
    
    try:
        from .nodes.nonlocal_means_node import NonLocalMeansNode
        NLM_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Non-local means node not available: {e}")
        NonLocalMeansNode = None
        NLM_AVAILABLE = False
    
    try:
        from .nodes.richardson_lucy_node import RichardsonLucyNode
        RL_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Richardson-Lucy node not available: {e}")
        RichardsonLucyNode = None
        RL_AVAILABLE = False

    try:
        from .nodes.richardson_lucy_gpu_node import RichardsonLucyGPUNode
        RL_GPU_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Richardson-Lucy GPU node not available: {e}")
        RichardsonLucyGPUNode = None
        RL_GPU_AVAILABLE = False
    
    try:
        from .nodes.wiener_filter_node import WienerFilterNode
        WIENER_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Wiener filter node not available: {e}")
        WienerFilterNode = None
        WIENER_AVAILABLE = False
    
    try:
        from .nodes.frequency_enhancement_node import (
            HomomorphicFilterNode,
            PhasePreservingEnhancementNode,
            MultiscaleFFTEnhancementNode,
            AdaptiveFrequencyFilterNode,
            FrequencyEnhancementPresetsNode
        )
        FREQ_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Frequency enhancement nodes not available: {e}")
        HomomorphicFilterNode = None
        PhasePreservingEnhancementNode = None
        MultiscaleFFTEnhancementNode = None
        AdaptiveFrequencyFilterNode = None
        FrequencyEnhancementPresetsNode = None
        FREQ_AVAILABLE = False
    try:
        from .nodes.adaptive_enhancement_node import AdaptiveImageEnhancementNode
        ADAPTIVE_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Adaptive enhancement node not available: {e}")
        AdaptiveImageEnhancementNode = None
        ADAPTIVE_AVAILABLE = False
    
    try:
        from .nodes.batch_processing_node import BatchImageProcessingNode
        BATCH_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Batch processing node not available: {e}")
        BatchImageProcessingNode = None
        BATCH_AVAILABLE = False
    
    try:
        from .nodes.quality_assessment_node import ImageQualityAssessmentNode
        QUALITY_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Quality assessment node not available: {e}")
        ImageQualityAssessmentNode = None
        QUALITY_AVAILABLE = False
    
    try:
        from .nodes.film_grain_node import FilmGrainProcessingNode, FilmGrainAnalysisNode
        from .nodes.advanced_film_grain_node import AdvancedFilmGrainNode
        FILM_GRAIN_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Film grain nodes not available: {e}")
        FilmGrainProcessingNode = None
        FilmGrainAnalysisNode = None
        AdvancedFilmGrainNode = None
        FILM_GRAIN_AVAILABLE = False
    
    # Import new film grain denoising nodes
    try:
        from .nodes.fga_nn_film_grain_node import FGANNFilmGrainDenoiseNode
        FGANN_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: FGA-NN film grain denoise node not available: {e}")
        FGANNFilmGrainDenoiseNode = None
        FGANN_AVAILABLE = False
    
    try:
        from .nodes.lightweight_cnn_denoise_node import LightweightCNNDenoiseNode
        PROGRESSIVE_CNN_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Lightweight Progressive CNN denoise node not available: {e}")
        LightweightCNNDenoiseNode = None
        PROGRESSIVE_CNN_AVAILABLE = False
    
    # Import pre-trained denoising nodes
    try:
        from .nodes.dncnn_node import DnCNNDenoiseNode
        DNCNN_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: DnCNN denoise node not available: {e}")
        DnCNNDenoiseNode = None
        DNCNN_AVAILABLE = False
    
    try:
        from .nodes.nafnet_node import NAFNetDenoiseNode
        NAFNET_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: NAFNet denoise node not available: {e}")
        NAFNetDenoiseNode = None
        NAFNET_AVAILABLE = False
    
    try:
        from .nodes.noise_da_node import NoiseDANode, NoiseDABatchNode
        NOISE_DA_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Noise-DA nodes not available: {e}")
        NoiseDANode = None
        NoiseDABatchNode = None
        NOISE_DA_AVAILABLE = False
    
    # Import advanced enhancement nodes
    try:
        from .nodes.advanced_enhancement_nodes import (
            LBCLAHENode,
            MultiScaleRetinexNode,
            BM3DFilmGrainNode,
            SmartSharpeningNode
        )
        ADVANCED_ENHANCEMENT_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Advanced enhancement nodes not available: {e}")
        LBCLAHENode = None
        MultiScaleRetinexNode = None
        BM3DFilmGrainNode = None
        SmartSharpeningNode = None
        ADVANCED_ENHANCEMENT_AVAILABLE = False
    
    # Import advanced AI nodes
    try:
        from .nodes.advanced_ai_nodes import (
            RealESRGANNode,
            PerceptualColorNode,
            AdvancedPSFNode,
            SFHformerNode,
            AIEnhancementBatchNode
        )
        ADVANCED_AI_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Advanced AI nodes not available: {e}")
        RealESRGANNode = None
        PerceptualColorNode = None
        AdvancedPSFNode = None
        SFHformerNode = None
        AIEnhancementBatchNode = None
        ADVANCED_AI_AVAILABLE = False
    
    # Import SCUNet nodes
    try:
        from .nodes.scunet_node import (
            SCUNetRestorationNode,
            SCUNetBatchRestorationNode
        )
        SCUNET_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: SCUNet nodes not available: {e}")
        SCUNetRestorationNode = None
        SCUNetBatchRestorationNode = None
        SCUNET_AVAILABLE = False
    
    # Import new advanced nodes
    try:
        from .nodes.swinir_node import (
            SwinIRRestorationNode,
            SwinIRBatchNode,
            MemoryOptimizationNode
        )
        from .nodes.swinir_sharpness_node import SwinIRSharpnessBoostNode
        from .nodes.restormer_node import RestormerRestorationNode
        from .nodes.diffbir_node import DiffBIRRestorationNode
        from .nodes.smart_workflow_node import SmartWorkflowNode
        from .nodes.professional_pipeline_node import ProfessionalRestorationPipelineNode
        from .nodes.comprehensive_comparison_node import ComprehensiveComparisonNode
        NEW_ADVANCED_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: New advanced nodes not available: {e}")
        SwinIRRestorationNode = None
        SwinIRBatchNode = None
        MemoryOptimizationNode = None
        SwinIRSharpnessBoostNode = None
        RestormerRestorationNode = None
        DiffBIRRestorationNode = None
        SmartWorkflowNode = None
        ProfessionalRestorationPipelineNode = None
        ComprehensiveComparisonNode = None
        NEW_ADVANCED_AVAILABLE = False
    
    # Import Auto-Denoise nodes
    try:
        from .nodes.auto_denoise_node import (
            AutoDenoiseNode,
            Noise2VoidNode,
            DeepImagePriorNode,
            AutoDenoiseComparisonNode
        )
        AUTO_DENOISE_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Auto-Denoise nodes not available: {e}")
        AutoDenoiseNode = None
        Noise2VoidNode = None
        DeepImagePriorNode = None
        AutoDenoiseComparisonNode = None
        AUTO_DENOISE_AVAILABLE = False
    
    # Import DeepInv nodes
    try:
        from .nodes.deepinv_denoise_node import DeepInvDenoiseNode
        DEEPINV_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: DeepInv nodes not available: {e}")
        DeepInvDenoiseNode = None
        DEEPINV_AVAILABLE = False

    # Import BM3D nodes
    try:
        from .nodes.bm3d_node import (
            BM3DDenoiseNode,
            BM3DDeblurNode,
            BM3DComparisonNode
        )
        BM3D_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: BM3D nodes not available: {e}")
        BM3DDenoiseNode = None
        BM3DDeblurNode = None
        BM3DComparisonNode = None
        BM3D_AVAILABLE = False
    
    # Import GPU BM3D node (optional, requires pytorch-bm3d)
    try:
        from .nodes.bm3d_gpu_denoise_node import BM3DGPUDenoiseNode
        from .scripts.bm3d_gpu_denoise import is_available
        gpu_available, gpu_reason = is_available()
        if gpu_available:
            BM3D_GPU_AVAILABLE = True
            print("GPU BM3D available - 15-30x faster than CPU BM3D!")
        else:
            BM3D_GPU_AVAILABLE = False
            print(f"GPU BM3D not available: {gpu_reason}")
    except ImportError as e:
        print(f"Note: GPU BM3D not available (optional): {e}")
        BM3DGPUDenoiseNode = None
        BM3D_GPU_AVAILABLE = False
    
    # Create node mappings (only for available nodes)
    WAVELET_MAPPINGS = {}
    WAVELET_DISPLAY = {}
    if WAVELET_AVAILABLE and WaveletDenoiseNode:
        WAVELET_MAPPINGS = {"WaveletDenoiseNode": WaveletDenoiseNode}
        WAVELET_DISPLAY = {"WaveletDenoiseNode": "Wavelet Denoise"}
    
    NLM_MAPPINGS = {}
    NLM_DISPLAY = {}
    if NLM_AVAILABLE and NonLocalMeansNode:
        NLM_MAPPINGS = {"NonLocalMeansNode": NonLocalMeansNode}
        NLM_DISPLAY = {"NonLocalMeansNode": "Non-Local Means Denoise"}
    
    RL_MAPPINGS = {}
    RL_DISPLAY = {}
    if RL_AVAILABLE and RichardsonLucyNode:
        RL_MAPPINGS["RichardsonLucyNode"] = RichardsonLucyNode
        RL_DISPLAY["RichardsonLucyNode"] = "Richardson-Lucy Deconvolution"
    if 'RL_GPU_AVAILABLE' in globals() and RL_GPU_AVAILABLE and RichardsonLucyGPUNode:
        RL_MAPPINGS["RichardsonLucyGPUNode"] = RichardsonLucyGPUNode
        RL_DISPLAY["RichardsonLucyGPUNode"] = "Richardson-Lucy Deconvolution GPU"
    
    WIENER_MAPPINGS = {}
    WIENER_DISPLAY = {}
    if WIENER_AVAILABLE and WienerFilterNode:
        WIENER_MAPPINGS = {"WienerFilterNode": WienerFilterNode}
        WIENER_DISPLAY = {"WienerFilterNode": "Wiener Filter Restoration"}
    
    FREQ_MAPPINGS = {}
    FREQ_DISPLAY = {}
    if FREQ_AVAILABLE:
        if HomomorphicFilterNode:
            FREQ_MAPPINGS["HomomorphicFilterNode"] = HomomorphicFilterNode
            FREQ_DISPLAY["HomomorphicFilterNode"] = "Homomorphic Filter"
        if PhasePreservingEnhancementNode:
            FREQ_MAPPINGS["PhasePreservingEnhancementNode"] = PhasePreservingEnhancementNode
            FREQ_DISPLAY["PhasePreservingEnhancementNode"] = "Phase-Preserving Enhancement"
        if MultiscaleFFTEnhancementNode:
            FREQ_MAPPINGS["MultiscaleFFTEnhancementNode"] = MultiscaleFFTEnhancementNode
            FREQ_DISPLAY["MultiscaleFFTEnhancementNode"] = "Multiscale FFT Enhancement"
        if AdaptiveFrequencyFilterNode:
            FREQ_MAPPINGS["AdaptiveFrequencyFilterNode"] = AdaptiveFrequencyFilterNode
            FREQ_DISPLAY["AdaptiveFrequencyFilterNode"] = "Adaptive Frequency Filter"
        if FrequencyEnhancementPresetsNode:
            FREQ_MAPPINGS["FrequencyEnhancementPresetsNode"] = FrequencyEnhancementPresetsNode
            FREQ_DISPLAY["FrequencyEnhancementPresetsNode"] = "Frequency Enhancement Presets"
    
    ADAPTIVE_MAPPINGS = {}
    ADAPTIVE_DISPLAY = {}
    if ADAPTIVE_AVAILABLE and AdaptiveImageEnhancementNode:
        ADAPTIVE_MAPPINGS = {"AdaptiveImageEnhancementNode": AdaptiveImageEnhancementNode}
        ADAPTIVE_DISPLAY = {"AdaptiveImageEnhancementNode": "Adaptive Image Enhancement"}
    
    BATCH_MAPPINGS = {}
    BATCH_DISPLAY = {}
    if BATCH_AVAILABLE and BatchImageProcessingNode:
        BATCH_MAPPINGS = {"BatchImageProcessingNode": BatchImageProcessingNode}
        BATCH_DISPLAY = {"BatchImageProcessingNode": "Batch Image Processing"}
    
    QUALITY_MAPPINGS = {}
    QUALITY_DISPLAY = {}
    if QUALITY_AVAILABLE and ImageQualityAssessmentNode:
        QUALITY_MAPPINGS = {"ImageQualityAssessmentNode": ImageQualityAssessmentNode}
        QUALITY_DISPLAY = {"ImageQualityAssessmentNode": "Image Quality Assessment"}
    
    FILM_GRAIN_MAPPINGS = {}
    FILM_GRAIN_DISPLAY = {}
    if FILM_GRAIN_AVAILABLE:
        if FilmGrainProcessingNode:
            FILM_GRAIN_MAPPINGS["FilmGrainProcessingNode"] = FilmGrainProcessingNode
            FILM_GRAIN_DISPLAY["FilmGrainProcessingNode"] = "Film Grain Processing"
        if FilmGrainAnalysisNode:
            FILM_GRAIN_MAPPINGS["FilmGrainAnalysisNode"] = FilmGrainAnalysisNode
            FILM_GRAIN_DISPLAY["FilmGrainAnalysisNode"] = "Film Grain Analysis"
        if AdvancedFilmGrainNode:
            FILM_GRAIN_MAPPINGS["AdvancedFilmGrainNode"] = AdvancedFilmGrainNode
            FILM_GRAIN_DISPLAY["AdvancedFilmGrainNode"] = "Advanced Film Grain Processing"
    
    # New film grain denoising mappings
    FGANN_MAPPINGS = {}
    FGANN_DISPLAY = {}
    if FGANN_AVAILABLE and FGANNFilmGrainDenoiseNode:
        FGANN_MAPPINGS = {"FGANNFilmGrainDenoiseNode": FGANNFilmGrainDenoiseNode}
        FGANN_DISPLAY = {"FGANNFilmGrainDenoiseNode": "FGA-NN Film Grain Denoise"}
        print("✓ FGA-NN Film Grain Denoise node loaded successfully")
    
    PROGRESSIVE_CNN_MAPPINGS = {}
    PROGRESSIVE_CNN_DISPLAY = {}
    if PROGRESSIVE_CNN_AVAILABLE and LightweightCNNDenoiseNode:
        PROGRESSIVE_CNN_MAPPINGS = {"LightweightCNNDenoiseNode": LightweightCNNDenoiseNode}
        PROGRESSIVE_CNN_DISPLAY = {"LightweightCNNDenoiseNode": "Lightweight Progressive CNN Denoise"}
        print("✓ Lightweight Progressive CNN Denoise node loaded successfully")
    
    # Pre-trained denoising node mappings
    DNCNN_MAPPINGS = {}
    DNCNN_DISPLAY = {}
    if DNCNN_AVAILABLE and DnCNNDenoiseNode:
        DNCNN_MAPPINGS = {"DnCNNDenoiseNode": DnCNNDenoiseNode}
        DNCNN_DISPLAY = {"DnCNNDenoiseNode": "DnCNN Denoise (Pre-trained)"}
        print("✓ DnCNN Denoise node loaded successfully")
    
    NAFNET_MAPPINGS = {}
    NAFNET_DISPLAY = {}
    if NAFNET_AVAILABLE and NAFNetDenoiseNode:
        NAFNET_MAPPINGS = {"NAFNetDenoiseNode": NAFNetDenoiseNode}
        NAFNET_DISPLAY = {"NAFNetDenoiseNode": "NAFNet Denoise (Pre-trained)"}
        print("✓ NAFNet Denoise node loaded successfully")
    
    # Noise-DA mappings
    NOISE_DA_MAPPINGS = {}
    NOISE_DA_DISPLAY = {}
    if NOISE_DA_AVAILABLE:
        if NoiseDANode:
            NOISE_DA_MAPPINGS["NoiseDANode"] = NoiseDANode
            NOISE_DA_DISPLAY["NoiseDANode"] = "Noise-DA Processing"
        if NoiseDABatchNode:
            NOISE_DA_MAPPINGS["NoiseDABatchNode"] = NoiseDABatchNode
            NOISE_DA_DISPLAY["NoiseDABatchNode"] = "Noise-DA Batch Processing"
    
    # Advanced enhancement mappings
    ADVANCED_MAPPINGS = {}
    ADVANCED_DISPLAY = {}
    if ADVANCED_ENHANCEMENT_AVAILABLE:
        if LBCLAHENode:
            ADVANCED_MAPPINGS["LBCLAHENode"] = LBCLAHENode
            ADVANCED_DISPLAY["LBCLAHENode"] = "LB-CLAHE (Learning-Based)"
        if MultiScaleRetinexNode:
            ADVANCED_MAPPINGS["MultiScaleRetinexNode"] = MultiScaleRetinexNode
            ADVANCED_DISPLAY["MultiScaleRetinexNode"] = "Multi-Scale Retinex"
        if BM3DFilmGrainNode:
            ADVANCED_MAPPINGS["BM3DFilmGrainNode"] = BM3DFilmGrainNode
            ADVANCED_DISPLAY["BM3DFilmGrainNode"] = "BM3D Film Grain Denoising"
        if SmartSharpeningNode:
            ADVANCED_MAPPINGS["SmartSharpeningNode"] = SmartSharpeningNode
            ADVANCED_DISPLAY["SmartSharpeningNode"] = "Smart Sharpening"
    
    # Advanced AI mappings
    ADVANCED_AI_MAPPINGS = {}
    ADVANCED_AI_DISPLAY = {}
    if ADVANCED_AI_AVAILABLE:
        if RealESRGANNode:
            ADVANCED_AI_MAPPINGS["RealESRGANNode"] = RealESRGANNode
            ADVANCED_AI_DISPLAY["RealESRGANNode"] = "Real-ESRGAN Super-Resolution"
        if PerceptualColorNode:
            ADVANCED_AI_MAPPINGS["PerceptualColorNode"] = PerceptualColorNode
            ADVANCED_AI_DISPLAY["PerceptualColorNode"] = "Perceptual Color Enhancement"
        if AdvancedPSFNode:
            ADVANCED_AI_MAPPINGS["AdvancedPSFNode"] = AdvancedPSFNode
            ADVANCED_AI_DISPLAY["AdvancedPSFNode"] = "Advanced PSF Modeling"
        if SFHformerNode:
            ADVANCED_AI_MAPPINGS["SFHformerNode"] = SFHformerNode
            ADVANCED_AI_DISPLAY["SFHformerNode"] = "SFHformer Dual-Domain"
        if AIEnhancementBatchNode:
            ADVANCED_AI_MAPPINGS["AIEnhancementBatchNode"] = AIEnhancementBatchNode
            ADVANCED_AI_DISPLAY["AIEnhancementBatchNode"] = "AI Enhancement Batch"
    
    # SCUNet mappings
    SCUNET_MAPPINGS = {}
    SCUNET_DISPLAY = {}
    if SCUNET_AVAILABLE:
        if SCUNetRestorationNode:
            SCUNET_MAPPINGS["SCUNetRestorationNode"] = SCUNetRestorationNode
            SCUNET_DISPLAY["SCUNetRestorationNode"] = "SCUNet Image Restoration"
        if SCUNetBatchRestorationNode:
            SCUNET_MAPPINGS["SCUNetBatchRestorationNode"] = SCUNetBatchRestorationNode
            SCUNET_DISPLAY["SCUNetBatchRestorationNode"] = "SCUNet Batch Processing"
    
    # New advanced node mappings
    NEW_ADVANCED_MAPPINGS = {}
    NEW_ADVANCED_DISPLAY = {}
    if NEW_ADVANCED_AVAILABLE:
        if SwinIRRestorationNode:
            NEW_ADVANCED_MAPPINGS["SwinIRRestorationNode"] = SwinIRRestorationNode
            NEW_ADVANCED_DISPLAY["SwinIRRestorationNode"] = "SwinIR Image Restoration"
        if SwinIRBatchNode:
            NEW_ADVANCED_MAPPINGS["SwinIRBatchNode"] = SwinIRBatchNode
            NEW_ADVANCED_DISPLAY["SwinIRBatchNode"] = "SwinIR Batch Processing"
        if SwinIRSharpnessBoostNode:
            NEW_ADVANCED_MAPPINGS["SwinIRSharpnessBoostNode"] = SwinIRSharpnessBoostNode
            NEW_ADVANCED_DISPLAY["SwinIRSharpnessBoostNode"] = "SwinIR Sharpness Boost"
        if RestormerRestorationNode:
            NEW_ADVANCED_MAPPINGS["RestormerRestorationNode"] = RestormerRestorationNode
            NEW_ADVANCED_DISPLAY["RestormerRestorationNode"] = "Restormer Restoration"
        if DiffBIRRestorationNode:
            NEW_ADVANCED_MAPPINGS["DiffBIRRestorationNode"] = DiffBIRRestorationNode
            NEW_ADVANCED_DISPLAY["DiffBIRRestorationNode"] = "DiffBIR Restoration (Eric)"
        if MemoryOptimizationNode:
            NEW_ADVANCED_MAPPINGS["MemoryOptimizationNode"] = MemoryOptimizationNode
            NEW_ADVANCED_DISPLAY["MemoryOptimizationNode"] = "Memory Optimization"
        if SmartWorkflowNode:
            NEW_ADVANCED_MAPPINGS["SmartWorkflowNode"] = SmartWorkflowNode
            NEW_ADVANCED_DISPLAY["SmartWorkflowNode"] = "Smart Workflow Selection"
        if ProfessionalRestorationPipelineNode:
            NEW_ADVANCED_MAPPINGS["ProfessionalRestorationPipelineNode"] = ProfessionalRestorationPipelineNode
            NEW_ADVANCED_DISPLAY["ProfessionalRestorationPipelineNode"] = "Professional Restoration Pipeline"
        if ComprehensiveComparisonNode:
            NEW_ADVANCED_MAPPINGS["ComprehensiveComparisonNode"] = ComprehensiveComparisonNode
            NEW_ADVANCED_DISPLAY["ComprehensiveComparisonNode"] = "Comprehensive Method Comparison"
    
    # Auto-Denoise node mappings
    AUTO_DENOISE_MAPPINGS = {}
    AUTO_DENOISE_DISPLAY = {}
    if AUTO_DENOISE_AVAILABLE:
        if AutoDenoiseNode:
            AUTO_DENOISE_MAPPINGS["AutoDenoiseNode"] = AutoDenoiseNode
            AUTO_DENOISE_DISPLAY["AutoDenoiseNode"] = "Auto-Denoise (Smart Selection)"
        if Noise2VoidNode:
            AUTO_DENOISE_MAPPINGS["Noise2VoidNode"] = Noise2VoidNode
            AUTO_DENOISE_DISPLAY["Noise2VoidNode"] = "Noise2Void (Self-Supervised)"
        if DeepImagePriorNode:
            AUTO_DENOISE_MAPPINGS["DeepImagePriorNode"] = DeepImagePriorNode
            AUTO_DENOISE_DISPLAY["DeepImagePriorNode"] = "Deep Image Prior (Unsupervised)"
        if AutoDenoiseComparisonNode:
            AUTO_DENOISE_MAPPINGS["AutoDenoiseComparisonNode"] = AutoDenoiseComparisonNode
            AUTO_DENOISE_DISPLAY["AutoDenoiseComparisonNode"] = "Auto-Denoise Comparison"

    # DeepInv node mappings
    DEEPINV_MAPPINGS = {}
    DEEPINV_DISPLAY = {}
    if DEEPINV_AVAILABLE:
        if DeepInvDenoiseNode:
            DEEPINV_MAPPINGS["DeepInvDenoiseNode"] = DeepInvDenoiseNode
            DEEPINV_DISPLAY["DeepInvDenoiseNode"] = "DeepInv Denoiser (Service)"
    
    # BM3D node mappings
    BM3D_MAPPINGS = {}
    BM3D_DISPLAY = {}
    if BM3D_AVAILABLE:
        if BM3DDenoiseNode:
            BM3D_MAPPINGS["BM3DDenoiseNode"] = BM3DDenoiseNode
            BM3D_DISPLAY["BM3DDenoiseNode"] = "BM3D Denoising"
        if BM3DDeblurNode:
            BM3D_MAPPINGS["BM3DDeblurNode"] = BM3DDeblurNode
            BM3D_DISPLAY["BM3DDeblurNode"] = "BM3D Deblurring"
        if BM3DComparisonNode:
            BM3D_MAPPINGS["BM3DComparisonNode"] = BM3DComparisonNode
            BM3D_DISPLAY["BM3DComparisonNode"] = "BM3D Profile Comparison"
    
    # GPU BM3D node mappings (optional, 15-30x faster than CPU)
    BM3D_GPU_MAPPINGS = {}
    BM3D_GPU_DISPLAY = {}
    if BM3D_GPU_AVAILABLE:
        if BM3DGPUDenoiseNode:
            BM3D_GPU_MAPPINGS["BM3DGPUDenoiseNode"] = BM3DGPUDenoiseNode
            BM3D_GPU_DISPLAY["BM3DGPUDenoiseNode"] = "BM3D GPU Denoise (Eric)"

except ImportError as e:
    print(f"Warning: Could not import some components: {e}")
    # Create empty mappings for missing components
    WAVELET_MAPPINGS = WAVELET_DISPLAY = {}
    NLM_MAPPINGS = NLM_DISPLAY = {}
    RL_MAPPINGS = RL_DISPLAY = {}
    WIENER_MAPPINGS = WIENER_DISPLAY = {}
    FREQ_MAPPINGS = FREQ_DISPLAY = {}
    ADAPTIVE_MAPPINGS = ADAPTIVE_DISPLAY = {}
    BATCH_MAPPINGS = BATCH_DISPLAY = {}
    QUALITY_MAPPINGS = QUALITY_DISPLAY = {}
    BM3D_GPU_MAPPINGS = BM3D_GPU_DISPLAY = {}
    FILM_GRAIN_MAPPINGS = FILM_GRAIN_DISPLAY = {}
    NOISE_DA_MAPPINGS = NOISE_DA_DISPLAY = {}
    ADVANCED_MAPPINGS = ADVANCED_DISPLAY = {}
    ADVANCED_AI_MAPPINGS = ADVANCED_AI_DISPLAY = {}
    SCUNET_MAPPINGS = SCUNET_DISPLAY = {}
    NEW_ADVANCED_MAPPINGS = NEW_ADVANCED_DISPLAY = {}
    AUTO_DENOISE_MAPPINGS = AUTO_DENOISE_DISPLAY = {}
    BM3D_MAPPINGS = BM3D_DISPLAY = {}
    ADVANCED_SHARPENING_MAPPINGS = ADVANCED_SHARPENING_DISPLAY = {}
    LEARNING_CLAHE_MAPPINGS = LEARNING_CLAHE_DISPLAY = {}
    ADAPTIVE_FREQUENCY_MAPPINGS = ADAPTIVE_FREQUENCY_DISPLAY = {}
    CUTTING_EDGE_COMPARISON_MAPPINGS = CUTTING_EDGE_COMPARISON_DISPLAY = {}
    FGANN_MAPPINGS = FGANN_DISPLAY = {}
    PROGRESSIVE_CNN_MAPPINGS = PROGRESSIVE_CNN_DISPLAY = {}
    DNCNN_MAPPINGS = DNCNN_DISPLAY = {}
    NAFNET_MAPPINGS = NAFNET_DISPLAY = {}
    DEEPINV_MAPPINGS = DEEPINV_DISPLAY = {}
    
    # Advanced Sharpening node mappings
    ADVANCED_SHARPENING_MAPPINGS = {}
    ADVANCED_SHARPENING_DISPLAY = {}
    try:
        # Try relative import first (for ComfyUI context)
        from .nodes.advanced_sharpening_node import (
            ADVANCED_SHARPENING_MAPPINGS as ASM,
            ADVANCED_SHARPENING_DISPLAY as ASD
        )
        ADVANCED_SHARPENING_MAPPINGS = ASM
        ADVANCED_SHARPENING_DISPLAY = ASD
        print(f"✅ Advanced Sharpening nodes loaded: {len(ADVANCED_SHARPENING_MAPPINGS)} nodes")
    except ImportError as e:
        # Fallback to absolute import
        try:
            from nodes.advanced_sharpening_node import (
                ADVANCED_SHARPENING_MAPPINGS as ASM,
                ADVANCED_SHARPENING_DISPLAY as ASD
            )
            ADVANCED_SHARPENING_MAPPINGS = ASM
            ADVANCED_SHARPENING_DISPLAY = ASD
            print(f"✅ Advanced Sharpening nodes loaded via fallback: {len(ADVANCED_SHARPENING_MAPPINGS)} nodes")
        except ImportError as e2:
            print(f"⚠️ Warning: Advanced Sharpening nodes not available: {e}")
            print(f"⚠️ Fallback also failed: {e2}")
            ADVANCED_SHARPENING_MAPPINGS = ADVANCED_SHARPENING_DISPLAY = {}

# Learning-Based CLAHE node mappings
LEARNING_CLAHE_MAPPINGS = {}
LEARNING_CLAHE_DISPLAY = {}
try:
    from .nodes.learning_based_clahe_node import (
        LearningBasedCLAHENode,
        SimpleLearningCLAHENode,
        AdvancedColorSpaceCLAHENode
    )
    LEARNING_CLAHE_MAPPINGS["LearningBasedCLAHENode"] = LearningBasedCLAHENode
    LEARNING_CLAHE_DISPLAY["LearningBasedCLAHENode"] = "[ML] Learning-Based CLAHE"
    LEARNING_CLAHE_MAPPINGS["SimpleLearningCLAHENode"] = SimpleLearningCLAHENode
    LEARNING_CLAHE_DISPLAY["SimpleLearningCLAHENode"] = "[SIMPLE] Simple Learning CLAHE"
    LEARNING_CLAHE_MAPPINGS["AdvancedColorSpaceCLAHENode"] = AdvancedColorSpaceCLAHENode
    LEARNING_CLAHE_DISPLAY["AdvancedColorSpaceCLAHENode"] = "[COLOR] Advanced Color Space CLAHE"
except ImportError as e:
    print(f"Warning: Learning-Based CLAHE nodes not available: {e}")
    LEARNING_CLAHE_MAPPINGS = LEARNING_CLAHE_DISPLAY = {}

# Adaptive Frequency Decomposition node mappings
ADAPTIVE_FREQUENCY_MAPPINGS = {}
ADAPTIVE_FREQUENCY_DISPLAY = {}
try:
    from .nodes.adaptive_frequency_decomposition_node import (
        AdaptiveFrequencyDecompositionNode,
        SimpleFrequencyEnhancementNode,
        FrequencyBandControlNode
    )
    ADAPTIVE_FREQUENCY_MAPPINGS["AdaptiveFrequencyDecompositionNode"] = AdaptiveFrequencyDecompositionNode
    ADAPTIVE_FREQUENCY_DISPLAY["AdaptiveFrequencyDecompositionNode"] = "[AFD] Adaptive Frequency Decomposition"
    ADAPTIVE_FREQUENCY_MAPPINGS["SimpleFrequencyEnhancementNode"] = SimpleFrequencyEnhancementNode
    ADAPTIVE_FREQUENCY_DISPLAY["SimpleFrequencyEnhancementNode"] = "[FREQ] Simple Frequency Enhancement"
    ADAPTIVE_FREQUENCY_MAPPINGS["FrequencyBandControlNode"] = FrequencyBandControlNode
    ADAPTIVE_FREQUENCY_DISPLAY["FrequencyBandControlNode"] = "[BAND] Frequency Band Control"
except ImportError as e:
    print(f"Warning: Adaptive Frequency Decomposition nodes not available: {e}")
    ADAPTIVE_FREQUENCY_MAPPINGS = ADAPTIVE_FREQUENCY_DISPLAY = {}

# Cutting-Edge Comparison node mappings
CUTTING_EDGE_COMPARISON_MAPPINGS = {}
CUTTING_EDGE_COMPARISON_DISPLAY = {}
try:
    from .nodes.cutting_edge_comparison_node import (
        CuttingEdgeEnhancementComparisonNode,
        CuttingEdgePipelineNode
    )
    CUTTING_EDGE_COMPARISON_MAPPINGS["CuttingEdgeEnhancementComparisonNode"] = CuttingEdgeEnhancementComparisonNode
    CUTTING_EDGE_COMPARISON_DISPLAY["CuttingEdgeEnhancementComparisonNode"] = "[COMP] Cutting-Edge Enhancement Comparison"
    CUTTING_EDGE_COMPARISON_MAPPINGS["CuttingEdgePipelineNode"] = CuttingEdgePipelineNode
    CUTTING_EDGE_COMPARISON_DISPLAY["CuttingEdgePipelineNode"] = "[PIPE] Cutting-Edge Enhancement Pipeline"
except ImportError as e:
    print(f"Warning: Cutting-Edge Comparison nodes not available: {e}")
    CUTTING_EDGE_COMPARISON_MAPPINGS = CUTTING_EDGE_COMPARISON_DISPLAY = {}

# Try to import Real BM3D nodes
try:
    from .nodes.real_bm3d_node import (
        REAL_BM3D_NODE_CLASS_MAPPINGS,
        REAL_BM3D_NODE_DISPLAY_NAME_MAPPINGS
    )
    REAL_BM3D_MAPPINGS = REAL_BM3D_NODE_CLASS_MAPPINGS
    REAL_BM3D_DISPLAY = REAL_BM3D_NODE_DISPLAY_NAME_MAPPINGS
    print("Real BM3D GPU nodes loaded successfully!")
except ImportError as e:
    print(f"Warning: Real BM3D nodes not available: {e}")
    REAL_BM3D_MAPPINGS = REAL_BM3D_DISPLAY = {}

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(WAVELET_MAPPINGS)
NODE_CLASS_MAPPINGS.update(NLM_MAPPINGS)
NODE_CLASS_MAPPINGS.update(RL_MAPPINGS)
NODE_CLASS_MAPPINGS.update(WIENER_MAPPINGS)
NODE_CLASS_MAPPINGS.update(FREQ_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ADAPTIVE_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BATCH_MAPPINGS)
NODE_CLASS_MAPPINGS.update(QUALITY_MAPPINGS)
NODE_CLASS_MAPPINGS.update(FILM_GRAIN_MAPPINGS)
NODE_CLASS_MAPPINGS.update(NOISE_DA_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ADVANCED_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ADVANCED_AI_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SCUNET_MAPPINGS)
NODE_CLASS_MAPPINGS.update(NEW_ADVANCED_MAPPINGS)
NODE_CLASS_MAPPINGS.update(AUTO_DENOISE_MAPPINGS)
NODE_CLASS_MAPPINGS.update(DEEPINV_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BM3D_MAPPINGS)
NODE_CLASS_MAPPINGS.update(BM3D_GPU_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ADVANCED_SHARPENING_MAPPINGS)
NODE_CLASS_MAPPINGS.update(LEARNING_CLAHE_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ADAPTIVE_FREQUENCY_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CUTTING_EDGE_COMPARISON_MAPPINGS)
NODE_CLASS_MAPPINGS.update(REAL_BM3D_MAPPINGS)
NODE_CLASS_MAPPINGS.update(FGANN_MAPPINGS)
NODE_CLASS_MAPPINGS.update(PROGRESSIVE_CNN_MAPPINGS)
NODE_CLASS_MAPPINGS.update(DNCNN_MAPPINGS)
NODE_CLASS_MAPPINGS.update(NAFNET_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(WAVELET_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(NLM_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(RL_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(WIENER_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(FREQ_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(ADAPTIVE_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(BATCH_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(QUALITY_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(FILM_GRAIN_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(NOISE_DA_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(ADVANCED_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(ADVANCED_AI_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(SCUNET_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(NEW_ADVANCED_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(AUTO_DENOISE_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(DEEPINV_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(BM3D_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(BM3D_GPU_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(ADVANCED_SHARPENING_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(LEARNING_CLAHE_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(ADAPTIVE_FREQUENCY_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(CUTTING_EDGE_COMPARISON_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(REAL_BM3D_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(FGANN_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(PROGRESSIVE_CNN_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(DNCNN_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(NAFNET_DISPLAY)

# Export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS",
    # Processing functions
    "homomorphic_filter",
    "phase_preserving_enhancement", 
    "multiscale_fft_enhancement",
    "adaptive_frequency_filter",
    "get_frequency_enhancement_presets",
    "wavelet_denoise",
    "wavelet_denoise_stationary",
    "gpu_wavelet_denoise",
    "gpu_wavelet_denoise_stationary",
    "estimate_noise_level",
    "get_available_wavelets",
    "nonlocal_means_denoise",
    "adaptive_nonlocal_means",
    "get_recommended_parameters",
    "richardson_lucy_deconvolution",
    "richardson_lucy_deconvolution_gpu",
    "get_blur_presets",
    "estimate_motion_blur",
    "create_motion_psf",
    "create_gaussian_psf",
    "wiener_filter_restoration",
    "adaptive_wiener_filter",
    "parametric_wiener_filter",
    "get_wiener_presets",
    "get_gpu_info",
    "gpu_memory_info",
    "cleanup_gpu_memory",
    "can_use_gpu",
    "gpu_gaussian_blur",
    "gpu_bilateral_filter",
    "gpu_non_local_means",
    "gpu_frequency_filter",
    "process_with_psf_modeling",
    "get_psf_presets",
    "process_with_perceptual_color",
    "get_perceptual_color_presets",
    "analyze_grain_type",
    "denoise_film_grain",
    "get_grain_processing_recommendations",
    # Processor classes
    "FilmGrainProcessor",
    "FilmGrainAnalyzer",
    "NoiseDAProcessor",
    "LBCLAHEProcessor",
    "MultiScaleRetinexProcessor",
    "BM3DGTADProcessor",
    "SmartSharpeningProcessor",
    "LearningBasedCLAHEProcessor",
    "AdaptiveFrequencyDecompositionProcessor",
    "AutoDenoiseProcessor",
    "Noise2VoidProcessor",
    "DeepImagePriorProcessor",
    "BM3DProcessor",
    "RealESRGANProcessor",
    "get_realesrgan_presets",
    "SFHformerProcessor",
    "get_sfhformer_presets",
    "SCUNetProcessor",
    "PracticalSCUNetProcessor",
    "SimplifiedSCUNetProcessor",
    "SwinIRProcessor",
    "RestormerProcessor",
    "RESTORMER_MODEL_CONFIGS",
    "DiffBIRConfig",
    "DiffBIRProcessor",
    "MemoryManager",
    "AdvancedPSFProcessor",
    "get_psf_presets",
    "AdvancedSharpeningProcessor",
    "PerceptualColorProcessor",
    "get_perceptual_color_presets",
    # Base class
    "BaseImageProcessingNode"
]

print("Eric's Image Processing Nodes loaded successfully!")
if NODE_CLASS_MAPPINGS:
    print(f"Available denoising nodes: {[k for k in NODE_CLASS_MAPPINGS.keys() if 'Denoise' in k or 'NonLocal' in k]}")
    print(f"Available restoration nodes: {[k for k in NODE_CLASS_MAPPINGS.keys() if 'Richardson' in k or 'Wiener' in k]}")
    print(f"Available frequency enhancement nodes: {[k for k in NODE_CLASS_MAPPINGS.keys() if 'Homomorphic' in k or 'Phase' in k or 'Multiscale' in k or 'Frequency' in k]}")
    print(f"Available utility nodes: {[k for k in NODE_CLASS_MAPPINGS.keys() if 'Adaptive' in k or 'Batch' in k or 'Quality' in k]}")
    print(f"Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
else:
    print("Warning: No nodes were loaded successfully")
