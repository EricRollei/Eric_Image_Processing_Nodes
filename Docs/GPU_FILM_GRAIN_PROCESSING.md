# GPU-Accelerated Film Grain Processing Documentation

## Overview

This document describes the advanced GPU-accelerated wavelet denoising and specialized film grain processing capabilities added to Eric's Image Processing Nodes for ComfyUI.

## New Features

### 1. GPU-Accelerated Wavelet Denoising

#### GPU Wavelet Denoising Functions
- `gpu_wavelet_denoise()` - GPU-accelerated standard wavelet denoising
- `gpu_wavelet_denoise_stationary()` - GPU-accelerated stationary wavelet denoising
- Automatic fallback to CPU when GPU is not available or beneficial

#### Key Improvements
- **CUDA Acceleration**: Uses CuPy for GPU computation when available
- **Memory Management**: Automatic GPU memory management and cleanup
- **Performance Optimization**: Only uses GPU for larger images where beneficial
- **Threshold Methods**: Enhanced BayesShrink, SureShrink, and VisuShrink implementations

### 2. Advanced Film Grain Analysis

#### FilmGrainAnalyzer Class
Comprehensive analysis of grain characteristics:

- **Noise Statistics**: Global and regional noise estimation
- **Grain Structure**: Size, regularity, and directionality analysis
- **Frequency Analysis**: Spectral characteristics and dominant frequencies
- **Texture Analysis**: Local texture features and GLCM properties
- **Automatic Classification**: Distinguishes between different grain types

#### Grain Types Detected
1. **Authentic Film Grain**: Natural film emulsion grain
2. **Digital Noise**: Camera sensor noise
3. **Simulated Overlay**: Artificially added grain effects
4. **Clean**: Low-noise images
5. **Mixed Grain**: Combination of multiple grain types

### 3. Specialized Film Grain Processing

#### FilmGrainProcessor Class
Tailored denoising strategies for each grain type:

#### Processing Strategies by Grain Type

**Authentic Film Grain**:
- Gentle wavelet denoising with BayesShrink
- Selective grain enhancement
- Gentle sharpening to preserve texture
- High preservation level (maintains film aesthetic)

**Digital Noise**:
- Non-local means denoising for main processing
- Wavelet denoising for residual noise
- Edge enhancement for lost details
- Aggressive noise reduction

**Simulated Overlay**:
- Frequency domain filtering to remove overlay
- Wavelet denoising for remaining artifacts
- Texture preservation from original
- Complete grain removal strategy

**Clean Images**:
- Gentle enhancement without aggressive processing
- Contrast improvement with CLAHE
- Detail preservation priority

**Mixed Grain**:
- Adaptive processing based on local characteristics
- Multi-stage approach combining multiple methods
- Balanced preservation and cleanup

### 4. ComfyUI Nodes

#### AdvancedFilmGrainNode
Complete film grain processing with automatic analysis:

**Input Parameters**:
- `processing_mode`: Auto-analyze or manual specification
- `grain_type`: Manual grain type selection
- `denoising_method`: GPU wavelet, stationary, CPU, or adaptive
- `preservation_level`: How much grain to preserve (0.0-1.0)
- `wavelet_type`: Wavelet family selection
- `thresholding_method`: BayesShrink, SureShrink, VisuShrink, manual
- `use_gpu`: Enable GPU acceleration
- `multi_stage_processing`: Enable multi-stage processing pipeline
- `edge_preservation`: Preserve edge information
- `grain_enhancement`: Enhance grain structure

**Outputs**:
- `processed_image`: Denoised image
- `grain_analysis`: Detailed grain analysis report
- `processing_info`: Processing method and performance information

#### GPUWaveletDenoiseNode
Standalone GPU-accelerated wavelet denoising:

**Features**:
- Direct GPU wavelet processing
- Stationary wavelet transform option
- Automatic CPU fallback
- Memory management
- Performance reporting

#### FilmGrainAnalysisNode
Diagnostic tool for grain analysis:

**Outputs**:
- `grain_type`: Detected grain type
- `analysis_summary`: Detailed analysis metrics
- `recommendations`: Suggested processing parameters

## Performance Optimizations

### GPU Memory Management
- Automatic memory estimation and safety checks
- Dynamic GPU/CPU selection based on image size
- Memory cleanup after processing
- Efficient memory usage patterns

### Processing Efficiency
- Only uses GPU for images > 512x512 pixels
- Batched processing for multiple images
- Optimal wavelet decomposition levels
- Smart threshold selection

### Fallback Mechanisms
- Automatic CPU fallback when GPU unavailable
- Graceful degradation on memory constraints
- Error handling with informative messages

## Usage Examples

### Basic GPU Wavelet Denoising
```python
from scripts.wavelet_denoise import gpu_wavelet_denoise

# Process image with GPU acceleration
denoised = gpu_wavelet_denoise(
    image,
    wavelet='db8',
    method='BayesShrink',
    mode='soft',
    levels=4,
    use_gpu=True
)
```

### Automatic Film Grain Processing
```python
from scripts.advanced_film_grain import process_film_grain

# Automatic grain analysis and processing
processed, analysis = process_film_grain(
    image,
    preservation_level=0.7,
    use_gpu=True
)

print(f"Detected grain type: {analysis['grain_type']}")
```

### Manual Grain Processing
```python
from scripts.advanced_film_grain import FilmGrainProcessor

processor = FilmGrainProcessor()

# Manual processing with specific grain type
processed, analysis = processor.process_grain_aware(
    image,
    auto_analyze=False,
    grain_type='authentic_film',
    preservation_level=0.8,
    use_gpu=True
)
```

## Best Practices

### For Authentic Film Grain
- Use preservation level 0.7-0.9
- Enable multi-stage processing
- Use BayesShrink thresholding
- Enable grain enhancement
- Wavelet: db8 or db4

### For Digital Noise
- Use preservation level 0.2-0.5
- Combine NLM with wavelet denoising
- Use SureShrink thresholding
- Enable edge preservation
- Wavelet: db4 or bior2.2

### For Simulated Overlays
- Use preservation level 0.0-0.3
- Apply frequency domain filtering
- Use VisuShrink thresholding
- Focus on texture preservation
- Wavelet: bior2.2 or coif2

## Technical Details

### GPU Acceleration Requirements
- **CuPy**: For CUDA acceleration
- **NVIDIA GPU**: CUDA-compatible graphics card
- **Memory**: Sufficient GPU memory for image processing
- **Fallback**: CPU processing when GPU unavailable

### Memory Requirements
- **Image Size**: Larger images benefit more from GPU processing
- **Memory Estimation**: Automatic memory usage calculation
- **Safety Margin**: 80% memory usage limit for stability

### Wavelet Transform Details
- **Standard DWT**: Fast, efficient, good for most cases
- **Stationary SWT**: Translation-invariant, better for structured content
- **Decomposition Levels**: Automatically calculated based on image size
- **Boundary Conditions**: Symmetric extension for artifact reduction

## Quality Metrics

### Evaluation Criteria
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Visual Quality**: Perceptual assessment
- **Grain Preservation**: Maintenance of desired grain characteristics

### Performance Benchmarks
- **GPU Speedup**: 2-5x faster than CPU for large images
- **Memory Efficiency**: Optimized memory usage patterns
- **Processing Time**: Sub-second processing for typical images

## Troubleshooting

### Common Issues

**GPU Not Available**:
- Check CuPy installation
- Verify CUDA drivers
- Ensure sufficient GPU memory
- Check GPU compatibility

**Memory Errors**:
- Reduce image size
- Close other GPU applications
- Increase system RAM
- Use CPU fallback

**Poor Results**:
- Verify grain type detection
- Adjust preservation level
- Try different wavelet types
- Check input image quality

### Error Messages
- "GPU transfer failed, using CPU": GPU memory issue
- "GPU wavelet denoising failed": Processing error with fallback
- "Insufficient GPU memory": Image too large for available memory

## Future Enhancements

### Planned Features
- **Additional Wavelets**: More wavelet families
- **Real-time Processing**: Live preview capabilities
- **Batch GPU Processing**: Multiple images simultaneously
- **Custom Grain Models**: User-defined grain characteristics
- **Advanced Metrics**: More sophisticated quality assessment

### Optimization Opportunities
- **Mixed Precision**: Half-precision for memory efficiency
- **Streaming Processing**: Large image tiling
- **Multi-GPU Support**: Distributed processing
- **Adaptive Algorithms**: Dynamic parameter adjustment

## Conclusion

The GPU-accelerated film grain processing system provides professional-quality denoising with specialized handling for different types of grain and noise. The automatic analysis ensures optimal processing parameters while maintaining the aesthetic qualities of the source material.

The system is designed for both novice users who want automatic processing and expert users who need fine-grained control over the denoising process. The GPU acceleration provides significant performance improvements while maintaining compatibility with CPU-only systems through automatic fallback mechanisms.
