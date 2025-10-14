#!/usr/bin/env python3
"""
Report Generation Module for Vehicle Inspection Pipeline
Uses open-source text generation models for AI-powered analysis
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
from io import BytesIO

try:
    # Try to import transformers for open-source text generation
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[ReportGenerator] Transformers not available. Install with: pip install transformers torch")

class VehicleInspectionReportGenerator:
    """
    Generates comprehensive vehicle inspection reports with AI analysis
    """
    
    def __init__(self, use_ai_analysis=True):
        self.use_ai_analysis = use_ai_analysis and TRANSFORMERS_AVAILABLE
        self.text_generator = None
        
        if self.use_ai_analysis:
            self._initialize_text_generator()
    
    def _initialize_text_generator(self):
        """Initialize open-source text generation model"""
        try:
            # Use a lightweight model that works well for technical analysis
            model_name = "microsoft/DialoGPT-medium"  # Good for conversational text
            # Alternative: "gpt2" for general text generation
            
            print("[ReportGenerator] Loading text generation model...")
            self.text_generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256  # GPT-2 pad token
            )
            print("[ReportGenerator] Text generation model loaded successfully")
            
        except Exception as e:
            print(f"[ReportGenerator] Failed to load text generation model: {e}")
            print("[ReportGenerator] Falling back to template-based descriptions")
            self.use_ai_analysis = False
    
    def generate_unique_report_id(self) -> str:
        """Generate unique report ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8].upper()
        return f"RPT_{timestamp}_{unique_id}"
    
    def generate_unique_defect_id(self) -> str:
        """Generate unique defect ID"""
        return f"D{uuid.uuid4().hex[:8].upper()}"
    
    def generate_defect_description(self, defect: Dict) -> str:
        """Generate AI-powered defect description"""
        if self.use_ai_analysis and self.text_generator:
            try:
                prompt = f"Vehicle defect analysis: {defect['class']} detected at {defect['location']} with {defect['score']:.1%} confidence. Technical assessment:"
                
                response = self.text_generator(
                    prompt,
                    max_length=len(prompt.split()) + 50,
                    num_return_sequences=1,
                    temperature=0.7
                )
                
                generated_text = response[0]['generated_text']
                # Extract only the generated part (after the prompt)
                description = generated_text[len(prompt):].strip()
                
                if description:
                    return description
                    
            except Exception as e:
                print(f"[ReportGenerator] AI generation failed: {e}")
        
        # Fallback to template-based descriptions
        return self._get_template_description(defect)
    
    def _get_template_description(self, defect: Dict) -> str:
        """Template-based defect descriptions"""
        defect_type = defect['class'].lower()
        location = defect['location']
        confidence = defect['score']
        
        templates = {
            'dent': f"Structural deformation detected in {location} area. "
                   f"Confidence level: {confidence:.1%}. "
                   f"Recommended action: Assess impact on structural integrity and consider repair.",
            
            'scratch': f"Surface scratch identified in {location} region. "
                      f"Detection confidence: {confidence:.1%}. "
                      f"Recommended action: Evaluate depth and extent for refinishing requirements."
        }
        
        return templates.get(defect_type, 
            f"{defect_type.title()} defect found in {location}. "
            f"Confidence: {confidence:.1%}. Requires further inspection.")
    
    def generate_overall_summary(self, defects: List[Dict], report_data: Dict) -> str:
        """Generate overall inspection summary"""
        total_defects = len(defects)
        defect_types = list(set([d['class'] for d in defects]))
        
        if self.use_ai_analysis and self.text_generator:
            try:
                prompt = f"Vehicle inspection summary: {total_defects} defects found including {', '.join(defect_types)}. Quality assessment:"
                
                response = self.text_generator(
                    prompt,
                    max_length=len(prompt.split()) + 60,
                    num_return_sequences=1,
                    temperature=0.6
                )
                
                generated_text = response[0]['generated_text']
                summary = generated_text[len(prompt):].strip()
                
                if summary:
                    return summary
                    
            except Exception as e:
                print(f"[ReportGenerator] AI summary generation failed: {e}")
        
        # Fallback template summary
        if total_defects == 0:
            return "No defects detected during inspection. Vehicle passes quality control standards."
        elif total_defects <= 2:
            return f"Minor quality issues detected ({total_defects} defects). Recommend monitoring and potential touch-up work."
        else:
            return f"Multiple defects identified ({total_defects} total). Requires comprehensive quality review and remediation."
    
    def create_pdf_report(self, report_data: Dict, defects: List[Dict], output_path: str = None) -> str:
        """Create comprehensive PDF report"""
        
        if output_path is None:
            output_path = f"reports/inspection_report_{report_data['id']}.pdf"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "reports", exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=20,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            spaceAfter=20
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkred,
            spaceBefore=15,
            spaceAfter=10
        )
        
        elements = []
        
        # Title
        elements.append(Paragraph("Vehicle Inspection Report", title_style))
        elements.append(Spacer(1, 20))
        
        # Report Information
        elements.append(Paragraph("Report Information", heading_style))
        report_info = [
            ["Report ID:", report_data['id']],
            ["Generated:", report_data['timestamp']],
            ["Image ID:", report_data.get('image_id', 'N/A')],
            ["Total Defects Found:", str(len(defects))]
        ]
        
        info_table = Table(report_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        # Vehicle Information
        elements.append(Paragraph("Vehicle Information", heading_style))
        vehicle_info = [
            ["Car Model:", "Standard Vehicle"],
            ["VIN:", f"VIN{uuid.uuid4().hex[:12].upper()}"],
            ["Inspection Date:", report_data['timestamp']],
            ["Inspector:", "AI Inspection System"]
        ]
        
        vehicle_table = Table(vehicle_info, colWidths=[2*inch, 4*inch])
        vehicle_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(vehicle_table)
        elements.append(Spacer(1, 20))
        
        # Defect Summary Table
        if defects:
            elements.append(Paragraph("Defect Summary Table", heading_style))
            
            table_data = [["Defect ID", "Type", "Location", "Confidence", "Severity", "Action Required"]]
            
            for defect in defects:
                severity = "High" if defect['score'] > 0.8 else "Medium" if defect['score'] > 0.6 else "Low"
                action = "Immediate Repair" if severity == "High" else "Monitor" if severity == "Low" else "Schedule Repair"
                
                table_data.append([
                    defect['id'],
                    defect['class'].title(),
                    defect['location'].title(),
                    f"{defect['score']:.1%}",
                    severity,
                    action
                ])
            
            defect_table = Table(table_data, colWidths=[1.2*inch, 1*inch, 1.3*inch, 1*inch, 1*inch, 1.5*inch])
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(defect_table)
            elements.append(Spacer(1, 20))
        
        # AI Analysis Section
        elements.append(Paragraph("AI Analysis", heading_style))
        
        if defects:
            for defect in defects:
                elements.append(Paragraph(f"Defect ID: {defect['id']}", styles['Heading3']))
                description = self.generate_defect_description(defect)
                elements.append(Paragraph(description, styles['Normal']))
                elements.append(Spacer(1, 10))
        
        # Overall Summary
        elements.append(Paragraph("Overall Assessment", heading_style))
        summary = self.generate_overall_summary(defects, report_data)
        elements.append(Paragraph(summary, styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Inspection Images Section
        if 'original_image_path' in report_data and 'processed_image_path' in report_data:
            elements.append(Paragraph("Inspection Images", heading_style))
            
            try:
                # Original image
                elements.append(Paragraph("Original Image", styles['Heading3']))
                if os.path.exists(report_data['original_image_path']):
                    img = ReportLabImage(report_data['original_image_path'], width=5*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 10))
                
                # Processed image with detections
                elements.append(Paragraph("Detection Results", styles['Heading3']))
                if os.path.exists(report_data['processed_image_path']):
                    img = ReportLabImage(report_data['processed_image_path'], width=5*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 10))
                    
            except Exception as e:
                elements.append(Paragraph(f"Error loading images: {str(e)}", styles['Normal']))
        
        # Build PDF
        try:
            doc.build(elements)
            print(f"[ReportGenerator] PDF report generated: {output_path}")
            return output_path
        except Exception as e:
            print(f"[ReportGenerator] Error generating PDF: {e}")
            return None
    
    def create_json_report(self, report_data: Dict, defects: List[Dict]) -> Dict:
        """Create JSON format report for API responses"""
        
        json_report = {
            "report_id": report_data['id'],
            "timestamp": report_data['timestamp'],
            "image_id": report_data.get('image_id', None),
            "total_defects": len(defects),
            "defects": defects,
            "summary": {
                "overall_assessment": self.generate_overall_summary(defects, report_data),
                "defect_types": list(set([d['class'] for d in defects])) if defects else [],
                "severity_distribution": self._calculate_severity_distribution(defects)
            },
            "ai_analysis": {
                "model_used": "Open Source Text Generation" if self.use_ai_analysis else "Template Based",
                "confidence_threshold": 0.5
            }
        }
        
        return json_report
    
    def _calculate_severity_distribution(self, defects: List[Dict]) -> Dict:
        """Calculate distribution of defect severities"""
        if not defects:
            return {"high": 0, "medium": 0, "low": 0}
        
        high = sum(1 for d in defects if d['score'] > 0.8)
        medium = sum(1 for d in defects if 0.6 < d['score'] <= 0.8)
        low = sum(1 for d in defects if d['score'] <= 0.6)
        
        return {"high": high, "medium": medium, "low": low}


# Example usage and testing
if __name__ == "__main__":
    # Test the report generator
    generator = VehicleInspectionReportGenerator()
    
    # Sample data
    sample_defects = [
        {
            "id": generator.generate_unique_defect_id(),
            "class": "Scratch",
            "score": 0.85,
            "location": "front panel",
            "bbox": [100, 150, 200, 250]
        },
        {
            "id": generator.generate_unique_defect_id(),
            "class": "Dent",
            "score": 0.72,
            "location": "rear door",
            "bbox": [300, 200, 400, 300]
        }
    ]
    
    sample_report_data = {
        "id": generator.generate_unique_report_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_id": "IMG_001"
    }
    
    # Generate reports
    json_report = generator.create_json_report(sample_report_data, sample_defects)
    print("JSON Report:", json.dumps(json_report, indent=2))
    
    pdf_path = generator.create_pdf_report(sample_report_data, sample_defects)
    print(f"PDF Report generated: {pdf_path}")
