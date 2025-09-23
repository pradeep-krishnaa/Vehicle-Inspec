# SAM2 Surface Defect Detection Demo

A standalone implementation of SAM2 (Segment Anything Model 2) for detecting surface defects in automotive and industrial applications.

## 🎯 Features

This implementation can detect various types of surface defects:

- **🎨 Paint Defects**: Color variations, uneven coating
- **🔍 Surface Contamination**: Spots, stains, foreign particles  
- **🦠 Corrosion/Rust**: Oxidation patterns, rust formation
- **💧 Water Spots**: Mineral deposits, water marks

## 📁 Files Included

1. **`sam2_surface_defect_detector.py`** - Main detection script
2. **`SAM2_Surface_Defect_Demo.ipynb`** - Google Colab notebook demo
3. **`example_usage.py`** - Simple command-line example
4. **`README_SAM2_Demo.md`** - This documentation

## 🚀 Quick Start for Google Colab

### Method 1: Using the Jupyter Notebook (Recommended)

1. **Upload to Colab**: Upload `SAM2_Surface_Defect_Demo.ipynb` to Google Colab
2. **Enable GPU**: Go to Runtime → Change runtime type → Hardware accelerator → GPU
3. **Run All Cells**: Execute cells in order to install dependencies and run detection
4. **Upload Images**: When prompted, upload your test images
5. **View Results**: See detected defects with visualizations

### Method 2: Using the Python Script

1. **Upload Files**: Upload both `sam2_surface_defect_detector.py` and your test image to Colab
2. **Install Dependencies**:
   ```python
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install git+https://github.com/facebookresearch/segment-anything-2.git
   !pip install opencv-python pillow matplotlib numpy tqdm
   ```
3. **Run Detection**:
   ```python
   from sam2_surface_defect_detector import SAM2SurfaceDefectDetector, visualize_results, print_detection_summary
   
   # Initialize detector
   detector = SAM2SurfaceDefectDetector()
   
   # Run detection
   detections = detector.detect_surface_defects("your_image.jpg")
   
   # Print results and visualize
   print_detection_summary(detections)
   visualize_results("your_image.jpg", detections, save_path="results.png")
   ```

## 💻 Local Usage

### Prerequisites

```bash
# Install PyTorch (choose appropriate version for your system)
pip install torch torchvision torchaudio

# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Install other dependencies
pip install opencv-python pillow matplotlib numpy tqdm
```

### Command Line Usage

```bash
# Simple detection
python sam2_surface_defect_detector.py --image_path "path/to/image.jpg"

# With custom SAM2 model
python sam2_surface_defect_detector.py --image_path "image.jpg" --sam2_model "facebook/sam2-hiera-small"

# Save visualization
python sam2_surface_defect_detector.py --image_path "image.jpg" --save_visualization "results.png"

# Limit to 1 detection (like your original single box result)
python sam2_surface_defect_detector.py --image_path "image.jpg" --max_detections 1

# Use example script
python example_usage.py "path/to/image.jpg"
```

### Programmatic Usage

```python
from sam2_surface_defect_detector import SAM2SurfaceDefectDetector

# Initialize detector
detector = SAM2SurfaceDefectDetector(
    sam2_model="facebook/sam2-hiera-tiny",  # or None for auto-selection
    device="cuda"  # or "cpu" or None for auto-detection
)

# Run detection
detections = detector.detect_surface_defects("image.jpg")

# Process results
for detection in detections:
    print(f"Found {detection['class']} with confidence {detection['score']:.3f}")
    print(f"Location: {detection['location']}")
    print(f"Bounding box: {detection['bbox']}")
```

## 🔧 Configuration Options

### SAM2 Model Options

- `facebook/sam2-hiera-tiny` - Fastest, good for demos
- `facebook/sam2-hiera-small` - Balanced speed/accuracy  
- `facebook/sam2-hiera-base-plus` - Better accuracy
- `facebook/sam2-hiera-large` - Best accuracy, slower

### Detection Parameters

You can control the number of detections returned:

```python
# Initialize with custom settings
detector = SAM2SurfaceDefectDetector(
    sam2_model="facebook/sam2-hiera-tiny",
    max_detections=1  # Return only the most significant defect
)
```

You can also modify detection thresholds in the `SAM2SurfaceDefectDetector` class:

```python
detector.defect_types = {
    'paint_defect': {'color_variance_threshold': 800, 'min_area': 100},
    'contamination': {'brightness_threshold': 30, 'min_area': 50},
    'corrosion': {'rust_hue_range': (10, 25), 'min_area': 75},
    'water_spots': {'circularity_threshold': 0.7, 'min_area': 25}
}
```

## 📊 Output Format

Each detection returns a dictionary with:

```python
{
    'class': 'Paint Defect',           # Human-readable class name
    'score': 0.85,                    # Confidence score (0-1)
    'defect_type': 'paint_defect',    # Internal defect type
    'bbox': [x1, y1, x2, y2],        # Bounding box coordinates
    'area': 1250,                     # Area in pixels
    'location': 'middle_center_area', # Location description
    'all_scores': {                   # Scores for all defect types
        'paint_defect': 0.85,
        'contamination': 0.23,
        # ...
    }
}
```

## 🎨 Visualization

The `visualize_results()` function creates a matplotlib plot showing:
- Original image
- Bounding boxes around detected defects
- Color-coded labels by defect type
- Confidence scores

Colors:
- 🔴 Red: Paint defects
- 🟠 Orange: Contamination  
- 🟤 Brown: Corrosion
- 🔵 Blue: Water spots

## ⚡ Performance Tips

### For Google Colab:
1. Use GPU runtime for faster processing
2. Start with `sam2-hiera-tiny` model for speed
3. Resize large images to reduce processing time

### For Local Use:
1. Ensure CUDA is properly installed for GPU acceleration
2. Use appropriate SAM2 model size based on your hardware
3. Consider batch processing for multiple images

## 🐛 Troubleshooting

### Common Issues:

**SAM2 Installation Failed:**
```bash
# Try installing with specific commit
pip install git+https://github.com/facebookresearch/segment-anything-2.git@main
```

**CUDA Out of Memory:**
- Use smaller SAM2 model (`sam2-hiera-tiny`)
- Reduce image resolution
- Process images one at a time

**No Defects Detected:**
- Check if image has sufficient contrast
- Adjust detection thresholds
- Try different SAM2 model sizes

**Import Errors:**
- Ensure all dependencies are installed
- Check Python version compatibility (3.8+)

## 📝 Example Results

The detector can identify:
- Scratches and paint chips on car surfaces
- Water spots on glass or metal
- Rust formation on metal components  
- Contamination spots on painted surfaces
- Color variations indicating paint defects

## 🔄 Integration

This detector can be integrated into:
- Quality control systems
- Automated inspection pipelines  
- Mobile apps for field inspection
- Web services for batch processing

## 📚 References

- [SAM2 Paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- [SAM2 GitHub Repository](https://github.com/facebookresearch/segment-anything-2)
- [Original InspecAI Pipeline](your-project-link-here)

## 📄 License

This demo is based on the original InspecAI Pipeline project. Please refer to the main project for licensing information.

---

**Happy Defect Detecting! 🔍✨**
