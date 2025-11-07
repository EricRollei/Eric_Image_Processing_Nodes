# Eric's Advanced Image Processing Nodes for ComfyUI

A comprehensive collection of advanced image processing nodes for ComfyUI, featuring state-of-the-art denoising, enhancement, and restoration techniques with GPU acceleration and specialized film grain processing.

## 🚀 Features

### **🎬 NEW: GPU-Accelerated Film Grain Processing**
- **Automatic Grain Analysis**: Intelligent detection of authentic film grain, digital noise, and simulated overlays
- **Specialized Processing**: Tailored denoising strategies for each grain type
- **GPU Acceleration**: CUDA-accelerated wavelet transforms for high-performance processing
- **Preservation Controls**: Fine-tuned grain preservation from 0% to 100%
- **Multi-stage Processing**: Advanced processing pipelines for optimal results


### **Denoising Algorithms**
- **GPU Wavelet Denoising**: Hardware-accelerated wavelet transforms with CuPy
  - BayesShrink, VisuShrink, SureShrink adaptive thresholding
  - Multiple wavelet families (Daubechies, Biorthogonal, Coiflets, etc.)
  - Regular and stationary (translation-invariant) wavelet transforms
  - Automatic CPU fallback for compatibility
- **Non-Local Means**: Texture-preserving denoising for natural images
- **Adaptive Enhancement**: Intelligent image analysis with automatic parameter optimization

### **Frequency Domain Processing**
- **Homomorphic Filtering**: Illumination and reflectance separation
- **Phase-Preserving Enhancement**: Frequency domain enhancement with natural results
- **Multi-scale FFT Enhancement**: Pyramidal frequency processing
- **Adaptive Frequency Filtering**: Content-aware frequency domain operations

### **Restoration Techniques**
- **Richardson-Lucy Deconvolution**: Advanced iterative deconvolution
- **Wiener Filtering**: Optimal noise-to-signal ratio restoration
- **Point Spread Function (PSF) Modeling**: Gaussian, motion, and custom PSF support

### **Advanced Features**
- **GPU Memory Management**: Intelligent memory usage with automatic cleanup
- **Batch Processing**: Efficient processing of multiple images
- **Adaptive Parameter Selection**: Automatic parameter optimization per image
- **Memory Management**: Optimized for large images and batch processing
- **Progress Tracking**: Real-time processing feedback
- **Comprehensive Error Handling**: Robust operation with fallback mechanisms

## 📦 Installation

1. **Clone or download** this repository to your ComfyUI custom nodes directory:
   ```
   ComfyUI/custom_nodes/Eric_Image_Processing_Nodes/
   ```

2. **Install dependencies** using your ComfyUI Python environment:
   ```bash
   # Navigate to the node directory
   cd ComfyUI/custom_nodes/Eric_Image_Processing_Nodes
   
   # Install requirements
   pip install -r requirements.txt
   ```

3. **Optional: Install CuPy for GPU acceleration**:
   ```bash
   # For CUDA 11.x
   pip install cupy-cuda11x
   
   # For CUDA 12.x
   pip install cupy-cuda12x
   ```

4. **Restart ComfyUI** to load the new nodes

## 📘 Documentation Highlights

- [DeepInv Service Setup & Usage Guide](./DEEPINV_SERVICE_GUIDE.md) – Run the FastAPI helper service and explore presets for the `DeepInv Denoise (Eric)` node.

## ️ Available Nodes

### **🎬 NEW: Film Grain Processing Nodes**

- `Advanced Film Grain Processing (Eric)` - Comprehensive grain analysis and specialized denoising
- `GPU Wavelet Denoising (Eric)` - Hardware-accelerated wavelet transforms with CuPy
- `Film Grain Analysis (Eric)` - Diagnostic tool for grain type detection and analysis

### **Denoising Nodes**

- `Wavelet Denoise (Eric)` - Advanced wavelet denoising with multiple methods
- `Stationary Wavelet Denoise (Eric)` - Translation-invariant wavelet denoising
- `Non-Local Means Denoise (Eric)` - Texture-preserving denoising
- `Adaptive Image Enhancement (Eric)` - Intelligent enhancement with content analysis
- `DeepInv Denoise (Eric)` - DiffUNet, RAM, SwinIR, and classical denoisers served by the external DeepInv service

**Note**: When recommendations mention "adaptive" denoising, this refers to the `Adaptive Image Enhancement (Eric)` node, which automatically analyzes your image and applies appropriate denoising techniques.

### **Frequency Enhancement Nodes**

- `Homomorphic Filter (Eric)` - Illumination correction and enhancement
- `Phase Preserving Enhancement (Eric)` - Natural frequency domain enhancement
- `Multiscale FFT Enhancement (Eric)` - Multi-resolution frequency processing
- `Adaptive Frequency Filter (Eric)` - Content-aware frequency filtering

### **Restoration Nodes**

- `Richardson-Lucy Deconvolution (Eric)` - Advanced deconvolution for blur removal
- `Wiener Filter Restoration (Eric)` - Optimal signal restoration

### **Utility Nodes**

- `Batch Image Processing (Eric)` - Efficient batch processing with consistent parameters

## 🎯 Usage Examples

### **Basic Denoising**

1. Load your noisy image
2. Add `Wavelet Denoise (Eric)` node
3. Connect image input
4. Adjust parameters:
   - **Wavelet Type**: `db8` for natural images, `haar` for simple images
   - **Method**: `BayesShrink` for adaptive thresholding
   - **Sigma**: Leave at 0 for auto-estimation
5. Process and view results

### **Illumination Correction**

1. Load image with uneven lighting
2. Add `Homomorphic Filter (Eric)` node
3. Adjust parameters:
   - **d0**: 30-60 for moderate correction
   - **gamma_h**: 1.5-2.5 for detail enhancement
   - **gamma_l**: 0.5-0.8 for illumination suppression
4. Process to see improved lighting

### **Batch Processing**

1. Load multiple images as a batch
2. Add `Batch Image Processing (Eric)` node
3. Select processing method (e.g., `wavelet_denoise`)
4. Configure parameters
5. Enable `show_progress` to monitor processing
6. Process entire batch efficiently

### **Adaptive Enhancement**

1. Load any image
2. Add `Adaptive Image Enhancement (Eric)` node
3. Set `auto_detect_type` to True
4. Choose enhancement level (conservative/moderate/aggressive)
5. Enable `show_analysis` to see content analysis
6. Process for intelligent enhancement

### **DeepInv Diffusion Denoising**

1. Start the external DeepInv service (see [guide](./DEEPINV_SERVICE_GUIDE.md)):

   ```powershell
   python -m external_tools.deepinv_service.service
   ```

1. Launch ComfyUI and add `DeepInv Denoise (Eric)`.
1. Pick a preset (DiffUNet, RAM, SwinIR, DRUNet, DnCNN, SCUNet).
1. Set `sigma` (0.0-0.5) for DiffUNet/RAM or leave at default for fixed-noise models.
1. Toggle `prefer_gpu` if CUDA resources are available.
1. Optionally target a remote instance by setting `service_url`.

### **🎬 Film Grain Processing**

1. **Automatic Processing**:
   - Load image with grain/noise
   - Add `Advanced Film Grain Processing (Eric)` node
   - Set `processing_mode` to "auto_analyze"
   - Adjust `preservation_level` (0.0-1.0)
   - Enable `use_gpu` for faster processing
   - Process to get automatically optimized results

2. **Manual Processing for Specific Grain Types**:
   - Set `processing_mode` to "manual_specify"
   - Choose appropriate `grain_type`:
     - **Authentic Film**: For real film grain (use preservation 0.7-0.9)
     - **Digital Noise**: For high-ISO camera noise (use preservation 0.2-0.5)
     - **Simulated Overlay**: For artificial grain effects (use preservation 0.0-0.3)
   - Select `denoising_method` (gpu_wavelet recommended)
   - Enable `multi_stage_processing` for best results

3. **GPU-Accelerated Wavelet Only**:
   - Add `GPU Wavelet Denoising (Eric)` node
   - Set `use_gpu` to True
   - Choose `wavelet_type` and `thresholding_method`
   - Enable `use_stationary` for translation-invariant processing

4. **Grain Analysis Only**:
   - Add `Film Grain Analysis (Eric)` node
   - Enable `detailed_analysis` for comprehensive metrics
   - Review grain type detection and recommendations

## 🎯 Best Practices for Film Grain Types

### **Authentic Film Grain** (Film Emulsion)
- **Recommended Settings**:
  - Grain Type: `authentic_film`
  - Preservation Level: `0.7-0.9`
  - Wavelet: `db8` (Daubechies 8)
  - Method: `BayesShrink`
  - Multi-stage: `True`
  - Grain Enhancement: `True`
- **Why**: Film grain has natural structure that adds character. BayesShrink preserves this while removing noise.

### **Digital Camera Noise** (High ISO)
- **Recommended Settings**:
  - Grain Type: `digital_noise`
  - Preservation Level: `0.2-0.5`
  - Denoising Method: `adaptive`
  - Wavelet: `db4` or `bior2.2`
  - Method: `SureShrink`
  - Edge Preservation: `True`
- **Why**: Digital noise is more uniform and less desirable. Aggressive removal with edge preservation maintains image quality.

### **Simulated Grain Overlays** (Added in Post)
- **Recommended Settings**:
  - Grain Type: `simulated_overlay`
  - Preservation Level: `0.0-0.3`
  - Denoising Method: `gpu_wavelet`
  - Wavelet: `bior2.2`
  - Method: `VisuShrink`
  - Multi-stage: `True`
- **Why**: Artificial overlays can be completely removed while preserving the underlying image texture.

### **Mixed Grain** (Multiple Sources)
- **Recommended Settings**:
  - Processing Mode: `auto_analyze`
  - Preservation Level: `0.5-0.7`
  - Denoising Method: `adaptive`
  - Multi-stage: `True`
- **Why**: Automatic analysis determines the best approach for complex grain combinations.

## ⚙️ Parameter Guide

### **Wavelet Denoising**
- **Wavelet Types**:
  - `db8`: Best all-around choice for natural images
  - `db4`: Faster, good for general use
  - `bior2.2`: Excellent for preserving edges
  - `haar`: Fastest, good for simple images

- **Thresholding Methods**:
  - `BayesShrink`: Adaptive, optimal for most images
  - `VisuShrink`: Conservative, preserves more detail
  - `SureShrink`: Hybrid approach using statistical estimation
  - `manual`: Use custom sigma value

- **Sigma Parameter (Manual Mode)**:
  - `0.1-2.0`: Ultra-light noise for high-quality images
  - `2.0-10.0`: Light noise
  - `10.0-25.0`: Moderate noise
  - `25.0-50.0`: Heavy noise
  - `50.0+`: Extreme noise

### **Non-Local Means**
- **h Parameter**: Filtering strength (auto-estimated if 0)
  - `0.1-1.0`: Ultra-light denoising for high-quality images
  - `1.0-5.0`: Subtle denoising, preserve fine details
  - `5.0-15.0`: Balanced denoising (recommended)
  - `15.0-30.0`: Strong denoising
  - `30.0+`: Very strong, may blur details
- **Patch Size**: 5-7 for fine textures, 7-11 for coarse textures
- **Search Window**: 11-15 standard, 17-50 for maximum quality

### **Richardson-Lucy Deconvolution**
- **Blur Size**: 
  - `0.1-0.5`: Ultra-fine corrections for high-res images
  - `0.5-1.5`: Light defocus, subtle sharpening
  - `1.5-4.0`: Moderate blur (recommended)
  - `4.0+`: Heavy blur
- **Motion Length**:
  - `0.5-2.0`: Micro-motion correction for high-res images
  - `2.0-10.0`: Light motion blur
  - `10.0-30.0`: Moderate motion (recommended)
  - `30.0+`: Heavy motion blur
- **Regularization**:
  - `0.0005-0.005`: Ultra-light smoothing for high-quality images
  - `0.005-0.02`: Light smoothing (recommended)
  - `0.02+`: Moderate to heavy smoothing

### **Wiener Filter**
- **Blur Size**:
  - `0.1-0.5`: Ultra-fine sharpening for high-res images
  - `0.5-2.0`: Light blur correction
  - `2.0-6.0`: Moderate blur (recommended)
  - `6.0+`: Heavy blur
- **K Value (Regularization)**:
  - `0.0001-0.001`: Ultra-clean images, maximum sharpening
  - `0.001-0.01`: Clean images, sharp results
  - `0.01-0.1`: Moderate noise (recommended)
  - `0.1+`: Noisy images, smooth results

### **Homomorphic Filtering**
- **d0**: Cutoff frequency (1-200)
  - `1.0-10.0`: Very strong illumination correction
  - `10.0-30.0`: Strong illumination correction
  - `30.0-60.0`: Moderate correction (recommended)
  - `60.0+`: Subtle correction
- **gamma_h**: High frequency gain (0.1-10.0)
  - `0.1-0.8`: Reduce high frequencies (smoothing)
  - `0.8-1.2`: Subtle detail enhancement
  - `1.2-2.5`: Enhance details (recommended)
  - `2.5+`: Strong to extreme detail enhancement
- **gamma_l**: Low frequency gain (0.01-2.0)
  - `0.01-0.3`: Very strong illumination suppression
  - `0.3-0.8`: Moderate suppression (recommended)
  - `0.8-1.0`: Gentle suppression
  - `1.0+`: Boost low frequencies

### **🎬 Film Grain Processing Parameters**
- **Processing Mode**:
  - `auto_analyze`: Automatic grain detection and processing
  - `manual_specify`: Manual grain type selection
- **Preservation Level**: How much grain to preserve
  - `0.0-0.3`: Aggressive denoising (remove most/all grain)
  - `0.3-0.7`: Moderate denoising (balanced approach)
  - `0.7-1.0`: Conservative denoising (preserve grain character)
- **Denoising Method**:
  - `gpu_wavelet`: GPU-accelerated wavelet (recommended for large images)
  - `gpu_stationary_wavelet`: Translation-invariant processing
  - `adaptive`: Grain-specific processing pipeline
- **Wavelet Selection for Grain Types**:
  - `db8`: Best for authentic film grain
  - `db4`: Good all-around choice, faster processing
  - `bior2.2`: Excellent for edge preservation (digital noise)
  - `coif2`: Good reconstruction properties
- **GPU Settings**:
  - `use_gpu`: Enable CUDA acceleration (requires CuPy)
  - Auto-enables for images > 512x512 pixels
  - 2-5x speedup on compatible hardware

## 🔧 Advanced Features

### **Memory Management**
- Automatic memory cleanup after processing
- Batch processing with optional memory-efficient mode
- GPU memory management for CUDA-enabled systems

### **Error Handling**
- Graceful fallback to original image on processing errors
- Comprehensive error messages with troubleshooting hints
- Optional fail-fast mode for batch processing

### **Performance Optimization**
- Efficient algorithms with O(N log N) complexity for frequency domain operations
- Optimized parameter selection to reduce computational overhead
- Multi-threading support where available

## 📊 Performance Benchmarks

### **Typical Processing Times (1920x1080 image)**
- Wavelet Denoising: 0.5-2 seconds
- Non-Local Means: 2-8 seconds
- Homomorphic Filtering: 0.3-1 second
- Richardson-Lucy: 1-10 seconds (depends on iterations)
- Batch Processing: ~1.2x single image time per image

### **Memory Usage**
- Peak memory: 2-4x input image size
- Batch processing: Configurable memory vs. speed tradeoff
- GPU acceleration: Significant speedup for large images

## 🐛 Troubleshooting

### **Common Issues**

1. **Import Error: "No module named 'pywt'"**
   - Solution: Install PyWavelets: `pip install PyWavelets`

2. **Out of Memory Error**
   - Enable memory-efficient mode in batch processing
   - Reduce batch size
   - Use smaller decomposition levels for wavelets

3. **Slow Processing**
   - Reduce iterations for Richardson-Lucy deconvolution
   - Use faster wavelet types (haar, db4)
   - Enable fast mode for Non-Local Means

4. **Poor Results**
   - Check image format (should be 0-255 range)
   - Adjust parameters based on image content
   - Use adaptive methods for automatic parameter selection

5. **GPU Acceleration Issues**
   - **"CuPy not available"**: Install CuPy with `pip install cupy-cuda12x`
   - **"GPU transfer failed"**: GPU memory insufficient, reduce image size or use CPU
   - **"CUDA out of memory"**: Close other GPU applications or reduce batch size

6. **Film Grain Processing Issues**
   - **Poor grain detection**: Try manual grain type specification
   - **Over-denoising**: Increase preservation level (0.7-0.9)
   - **Under-denoising**: Decrease preservation level (0.3-0.5)
   - **Artifacts**: Enable multi-stage processing and edge preservation

## 🔬 Algorithm Details

### **Wavelet Denoising**
- Uses PyWavelets 1.6.0 with advanced thresholding
- Implements BayesShrink: `threshold = σ²/√(max(0, variance - σ²))`
- Supports both regular DWT and stationary SWT transforms

### **Non-Local Means**
- Based on Buades-Coll-Morel algorithm
- Automatic parameter estimation using noise variance
- Optimized patch comparison with early termination

### **Homomorphic Filtering**
- Separates illumination and reflectance: `I(x,y) = L(x,y) × R(x,y)`
- Applies frequency domain filtering: `H(u,v) = (γH - γL)[1 - G(u,v)] + γL`
- Supports Gaussian and Butterworth filter shapes

### **🎬 Film Grain Processing**
- **Multi-Domain Analysis**: Combines noise statistics, frequency analysis, texture features, and grain structure
- **Grain Classification**: Uses decision tree based on high-frequency ratio, grain regularity, and SNR
- **Adaptive Processing**: Tailors denoising strategy based on detected grain type:
  - **Authentic Film**: BayesShrink wavelet + selective grain enhancement
  - **Digital Noise**: Non-local means + SureShrink wavelet + edge enhancement
  - **Simulated Overlay**: Frequency filtering + VisuShrink wavelet + texture preservation
- **GPU Acceleration**: Uses CuPy for wavelet coefficient thresholding and variance calculations
- **Memory Optimization**: Processes coefficients on GPU, reconstructs on CPU to minimize memory usage

## 📚 References

1. Donoho, D.L., & Johnstone, I.M. (1995). "Adapting to Unknown Smoothness via Wavelet Shrinkage"
2. Buades, A., Coll, B., & Morel, J.M. (2005). "A Non-Local Algorithm for Image Denoising"
3. Gonzalez, R.C., & Woods, R.E. (2017). "Digital Image Processing"
4. Richardson, W.H. (1972). "Bayesian-Based Iterative Method of Image Restoration"
5. Wiener, N. (1949). "Extrapolation, Interpolation, and Smoothing of Stationary Time Series"
6. Chang, S.G., Yu, B., & Vetterli, M. (2000). "Adaptive Wavelet Thresholding for Image Denoising and Compression"
7. Portilla, J., Strela, V., Wainwright, M.J., & Simoncelli, E.P. (2003). "Image Denoising Using Scale Mixtures of Gaussians in the Wavelet Domain"
8. Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K. (2008). "Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data"

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support, please open an issue on the GitHub repository with:
- Your ComfyUI version
- Error messages (if any)
- Sample images (if relevant)
- Processing parameters used

---

**Made with ❤️ for the ComfyUI community**
