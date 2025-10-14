
import os
import uuid
import cv2
import base64
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from app.detector.model import VehicleInspector
from app.utils.visualize import draw_inspection_results
from app.utils.schema import to_inspection_response_schema
from app.report_generator import VehicleInspectionReportGenerator

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "app/static/uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "app/static/outputs")

print("Defect Detection Pipeline initialized")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize vehicle inspector (includes defect detection + surface defect detection)
inspector = VehicleInspector(
    defect_model=os.environ.get("MODEL_TYPE", "fasterrcnn"),  # "fasterrcnn" as default
    model_path=os.environ.get("MODEL_PATH", "fasterrcnn_model.pth"),
    sam2_model=os.environ.get("SAM2_MODEL", "facebook/sam2-hiera-tiny")  # SAM2 model from HuggingFace
)

# Initialize report generator
report_generator = VehicleInspectionReportGenerator()

# Global camera object
camera = None

# Ensure reports directory exists
os.makedirs("app/static/reports", exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def get_camera():
    """Initialize camera if not already done"""
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                # Try to set camera properties with error handling
                try:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print("Camera initialized with custom properties")
                except Exception as prop_error:
                    print(f"Warning: Could not set camera properties: {prop_error}")
                    print("Camera initialized with default properties")
                
                # Test if camera can actually read frames
                ret, test_frame = camera.read()
                if ret:
                    print("Camera test successful")
                else:
                    print("Camera test failed - cannot read frames")
                    camera.release()
                    camera = None
                    return None
            else:
                print("Failed to open camera")
                camera = None
                return None
        except Exception as e:
            print(f"Camera initialization error: {e}")
            camera = None
            return None
    return camera

def generate_frames():
    """Generate camera frames for streaming"""
    global camera
    
    print("Starting camera stream...")
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            camera = get_camera()
            if camera is None:
                print(f"Camera not available for streaming (attempt {retry_count + 1})")
                retry_count += 1
                continue
            
            while True:
                try:
                    success, frame = camera.read()
                    if not success:
                        print("Failed to read frame from camera")
                        break
                    
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        print("Failed to encode frame")
                        break
                        
                except Exception as frame_error:
                    print(f"Frame processing error: {frame_error}")
                    break
                    
        except Exception as e:
            print(f"Error in camera streaming: {e}")
            
        # Clean up and retry
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None
            
        retry_count += 1
        print(f"Retrying camera initialization ({retry_count}/{max_retries})")
    
    print("Camera streaming stopped after max retries")

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/status')
def camera_status():
    """Check camera availability"""
    test_camera = cv2.VideoCapture(0)
    if test_camera.isOpened():
        test_camera.release()
        return jsonify({"status": "available", "message": "Camera is ready"})
    else:
        return jsonify({"status": "unavailable", "message": "Camera not found"}), 404

@app.route('/api/capture', methods=['POST'])
def capture_and_infer():
    """Capture image from camera and run inference"""
    # Use a separate camera instance for capture to avoid conflicts with streaming
    capture_camera = None
    try:
        capture_camera = cv2.VideoCapture(0)
        if not capture_camera.isOpened():
            return jsonify({"error": "Camera not available"}), 500
        
        # Try to set camera properties (with error handling)
        try:
            capture_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as prop_error:
            print(f"Warning: Could not set capture camera properties: {prop_error}")
        
        # Capture frame from camera
        success, frame = capture_camera.read()
        if not success:
            return jsonify({"error": "Failed to capture image"}), 500
        
        print(f"Captured frame with shape: {frame.shape}")
        
    except Exception as e:
        print(f"Capture error: {e}")
        return jsonify({"error": f"Camera capture failed: {str(e)}"}), 500
    finally:
        # Always release the capture camera
        if capture_camera is not None:
            try:
                capture_camera.release()
            except:
                pass
    
    # Get detection mode (default to FasterRCNN only)
    detection_mode = request.json.get('detection_mode', 'fasterrcnn_only') if request.is_json else request.form.get('detection_mode', 'fasterrcnn_only')
    
    # Generate unique IDs
    image_id = str(uuid.uuid4())[:8]
    report_id = report_generator.generate_unique_report_id()
    
    # Save captured frame
    in_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    out_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")
    cv2.imwrite(in_path, frame)
    
    # Run inference based on selected mode
    if detection_mode == 'fasterrcnn_only':
        # FasterRCNN only - just defect detection
        raw_defects = inspector.defect_detector.detect(in_path)
        inspection_results = {
            "defects": raw_defects,
            "surface_defects": []  # No surface defects in FasterRCNN-only mode
        }
        print(f"[API] FasterRCNN-only mode: Found {len(raw_defects)} defects")
    else:
        # Full Vehicle Inspection (FasterRCNN + SAM2)
        inspection_results = inspector.inspect(in_path)
        raw_defects = inspection_results.get('defects', [])
        print(f"[API] Full inspection mode: Found {len(raw_defects)} structural defects, {len(inspection_results.get('surface_defects', []))} surface defects")
    
    # Add unique IDs to each defect and prepare for report
    processed_defects = []
    for defect in raw_defects:
        defect_with_id = {
            "id": report_generator.generate_unique_defect_id(),
            "class": defect.get("class", "Unknown"),
            "score": defect.get("score", 0.0),
            "bbox": defect.get("bbox", [0, 0, 0, 0]),
            "location": _determine_location_from_bbox(defect.get("bbox", [0, 0, 0, 0]), frame.shape)
        }
        processed_defects.append(defect_with_id)
    
    # Visualization
    draw_inspection_results(in_path, inspection_results, out_path)
    
    # Generate report
    report_data = {
        "id": report_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_id": image_id,
        "original_image_path": in_path,
        "processed_image_path": out_path,
        "detection_mode": detection_mode
    }
    
    # Generate PDF report
    pdf_report_path = f"app/static/reports/report_{report_id}.pdf"
    pdf_path = report_generator.create_pdf_report(report_data, processed_defects, pdf_report_path)
    
    # Generate JSON report
    json_report = report_generator.create_json_report(report_data, processed_defects)
    
    # Prepare response
    response = to_inspection_response_schema(image_id=image_id, results=inspection_results, visualization_path=out_path)
    
    # Add report information to response
    response.update({
        "report_id": report_id,
        "report_pdf_url": f"/reports/{os.path.basename(pdf_path)}" if pdf_path else None,
        "report_json": json_report,
        "total_defects": len(processed_defects),
        "defects_with_ids": processed_defects
    })
    
    return jsonify(response)

def _determine_location_from_bbox(bbox, image_shape):
    """Determine location description from bounding box"""
    if len(bbox) != 4:
        return "unknown location"
    
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    
    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Determine horizontal position
    if center_x < width / 3:
        h_pos = "left"
    elif center_x < 2 * width / 3:
        h_pos = "center"
    else:
        h_pos = "right"
    
    # Determine vertical position
    if center_y < height / 3:
        v_pos = "top"
    elif center_y < 2 * height / 3:
        v_pos = "middle"
    else:
        v_pos = "bottom"
    
    return f"{v_pos} {h_pos}"


@app.route("/outputs/<filename>")
def outputs(filename):
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"[DEBUG] Serving output file: {filename} from {abs_output_dir}")
    file_path = os.path.join(abs_output_dir, filename)
    print(f"[DEBUG] Full file path: {file_path}")
    print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
    return send_from_directory(abs_output_dir, filename, as_attachment=False)

@app.route("/uploads/<filename>")
def uploads(filename):
    abs_upload_dir = os.path.abspath(UPLOAD_DIR)
    print(f"[DEBUG] Serving upload file: {filename} from {abs_upload_dir}")
    file_path = os.path.join(abs_upload_dir, filename)
    print(f"[DEBUG] Full file path: {file_path}")
    print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
    return send_from_directory(abs_upload_dir, filename, as_attachment=False)

@app.route("/reports/<filename>")
def reports(filename):
    """Serve PDF reports"""
    reports_dir = os.path.abspath("app/static/reports")
    print(f"[DEBUG] Serving report file: {filename} from {reports_dir}")
    file_path = os.path.join(reports_dir, filename)
    print(f"[DEBUG] Report file exists: {os.path.exists(file_path)}")
    return send_from_directory(reports_dir, filename, as_attachment=False)

@app.teardown_appcontext
def cleanup_camera(error):
    """Clean up camera resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
    finally:
        # Cleanup camera on exit
        if camera is not None:
            camera.release()
